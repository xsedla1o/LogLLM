import os.path
import torch
import numpy as np
from torch import nn
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    DynamicCache,
)
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

memory_debug = int(os.getenv("MEMORY_DEBUG", "0"))


def merge_data(data):
    merged_data = []
    start_positions = []
    current_position = 0

    for sublist in data:
        start_positions.append(current_position)
        merged_data.extend(sublist)
        current_position += len(sublist)
    return merged_data, start_positions


def stack_and_pad_right(tensors):
    max_len = max(tensor.shape[0] for tensor in tensors)
    padded_tensors, padding_masks = [], []

    for tensor in tensors:
        pad_len = max_len - tensor.shape[0]
        padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_len))
        padded_tensors.append(padded_tensor)
        padding_mask = torch.cat(
            [
                torch.ones(tensor.shape[0], dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ]
        )
        padding_masks.append(padding_mask)

    stacked_tensor = torch.stack(padded_tensors)
    padding_masks = torch.stack(padding_masks)
    return stacked_tensor, padding_masks


def stack_and_pad_left(tensors):
    max_len = max(tensor.shape[0] for tensor in tensors)
    padded_tensors, padding_masks = [], []

    for tensor in tensors:
        pad_len = max_len - tensor.shape[0]
        padded_tensor = torch.nn.functional.pad(tensor, (0, 0, pad_len, 0))
        padded_tensors.append(padded_tensor)
        padding_mask = torch.cat(
            [
                torch.zeros(pad_len, dtype=torch.long),
                torch.ones(tensor.shape[0], dtype=torch.long),
            ]
        )
        padding_masks.append(padding_mask)

    stacked_tensor = torch.stack(padded_tensors)
    padding_masks = torch.stack(padding_masks)
    return stacked_tensor, padding_masks


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


class LogLLM(nn.Module):
    def __init__(
        self,
        Llama_path,
        ft_path=None,
        is_train_mode=True,
        device=torch.device("cuda:0"),
        max_content_len=128,
        max_seq_len=128,
        embed_pooling="none",
    ):
        super().__init__()
        self.max_content_len = max_content_len
        self.max_seq_len = max_seq_len
        self.device = device
        self.embed_pooling = embed_pooling

        self.Llama_tokenizer = AutoTokenizer.from_pretrained(
            Llama_path, padding_side="right"
        )
        self.Llama_tokenizer.pad_token = self.Llama_tokenizer.eos_token
        self.Llama_model = AutoModelForCausalLM.from_pretrained(
            Llama_path,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
        # Gradient checkpointing for memory savings
        self.Llama_model.gradient_checkpointing_enable()
        self.Llama_model.config.use_cache = False

        self.instruc_tokens = self.Llama_tokenizer(
            [
                "Below is a sequence of system log messages:",
                ". Is this sequence normal or anomalous?\n",
            ],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        if ft_path is not None:
            Llama_ft_path = self._get_ft_model_path(ft_path)
            print(f"Loading fine-tuned model from {ft_path}.")
            self.Llama_model = PeftModel.from_pretrained(
                self.Llama_model,
                Llama_ft_path,
                is_trainable=is_train_mode,
                torch_dtype=torch.float16,
            )
        else:
            print("Creating new PEFT model.")
            Llama_peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.Llama_model = get_peft_model(self.Llama_model, Llama_peft_config)
        # Enable gradients for input embeddings, required for gradient checkpointing
        self.Llama_model.enable_input_require_grads()

        # Precompute fixed instruction/prefix lengths for context budgeting
        with torch.no_grad():
            # These embeddings don’t change, so we only need their token counts
            ins1_len = int(self.instruc_tokens["attention_mask"][0].sum())
            ins2_len = (
                int(self.instruc_tokens["attention_mask"][1].sum()) - 1
            )  # skip first token
            prefix = "The sequence is"
            prefix_ids = self.Llama_tokenizer(prefix, return_tensors="pt")["input_ids"][
                0, 1:
            ]
            prefix_len = prefix_ids.shape[0]

        self.fixed_prompt_len = ins1_len + ins2_len + prefix_len
        self.generation_budget = 64  # reserve space for generated tokens
        self.context_limit = 8192
        self.available_ctx = (
            self.context_limit - self.fixed_prompt_len - self.generation_budget
        )
        print(
            f"[Info] Context budgeting [tokens]: "
            f"fixed={self.fixed_prompt_len}, generation={self.generation_budget}, "
            f"available={self.available_ctx}"
        )

    def _get_ft_model_path(self, path):
        full_path = os.path.join(path, str(self.embed_pooling))
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        Llama_ft_path = os.path.join(full_path, "Llama_ft")
        return Llama_ft_path

    def save_ft_model(self, path):
        Llama_ft_path = self._get_ft_model_path(path)
        self.Llama_model.save_pretrained(Llama_ft_path, safe_serialization=True)

    def set_train_only_Llama(self):
        for name, param in self.Llama_model.named_parameters():
            param.requires_grad = "lora" in name

    def set_finetuning_all(self):
        for name, param in self.Llama_model.named_parameters():
            if "lora" in name:
                param.requires_grad = True

    def _get_embed_tokens(self):
        # Works for base Llama or PEFT-wrapped
        if hasattr(self.Llama_model, "model") and hasattr(
            self.Llama_model.model, "model"
        ):
            return self.Llama_model.model.model.embed_tokens
        return self.Llama_model.model.embed_tokens

    def _llama_embed_tokens(self, inputs, seq_positions, pooling="mean"):
        device = self.device

        # Per-line mean-pooled embeddings
        embed_tokens = self._get_embed_tokens()
        tok_emb = embed_tokens(inputs["input_ids"].to(device))  # [N_lines, T, H]
        att = inputs["attention_mask"].to(device).bool()  # [N_lines, T]
        lengths = att.sum(dim=1).clamp(min=1).unsqueeze(-1)  # [N_lines, 1]

        if pooling == "mean":
            # [N_lines, H]
            line_embeds = (tok_emb * att.unsqueeze(-1)).sum(dim=1) / lengths
            # Split into sequences, list of [L_i, H]
            seq_embeddings = torch.tensor_split(line_embeds, seq_positions)
        elif pooling == "max":
            tok_emb = tok_emb.masked_fill(~att.unsqueeze(-1), float("-inf"))
            line_embeds, _ = tok_emb.max(dim=1)  # [N_lines, H]
            seq_embeddings = torch.tensor_split(line_embeds, seq_positions)
        elif pooling is None or pooling == "none":
            # Keep only non-padded tokens per sequence
            line_groups = torch.tensor_split(tok_emb, seq_positions)
            mask_groups = torch.tensor_split(att, seq_positions)

            seq_embeddings = []
            token_limit = self.available_ctx
            for lines, masks in zip(line_groups, mask_groups):
                seq_emb = lines[masks]  # flatten valid tokens only

                if seq_emb.shape[0] > token_limit:
                    overflow = seq_emb.shape[0] - token_limit
                    seq_emb = seq_emb[-token_limit:]  # keep tail tokens
                    print(
                        f"[Truncate] Sequence trimmed by {overflow} tokens "
                        f"(limit {token_limit})",
                    )

                seq_embeddings.append(seq_emb)
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        return seq_embeddings

    def train_helper(self, inputs, seq_positions, labels):
        device = self.device
        batch_size = len(labels)

        # ---- 1) Per-line mean-pooled embeddings from Llama's token embeddings ----
        embed_tokens = self._get_embed_tokens()
        seq_embeddings = self._llama_embed_tokens(
            inputs, seq_positions, pooling=self.embed_pooling
        )  # list of [N_i, H]

        # ---- 2) Build targets from labels (same text format) ----
        prefix = "The sequence is "
        max_len = max(len(s) for s in labels) + len(prefix)
        labels = np.char.add(np.char.add(prefix, labels.astype(f"U{max_len}")), ".")
        answer_tokens = self.Llama_tokenizer(
            list(labels), padding=True, return_tensors="pt"
        ).to(device)

        # Create IDs and attention mask with EOS
        ans_ids_wo_bos = answer_tokens["input_ids"][:, 1:]
        ans_mask_wo_bos = answer_tokens["attention_mask"][:, 1:].bool()
        eos_col = torch.ones((batch_size, 1), dtype=torch.bool, device=device)

        target_tokens_ids = torch.cat(
            [
                ans_ids_wo_bos,
                torch.full(
                    (batch_size, 1), self.Llama_tokenizer.eos_token_id, device=device
                ),
            ],
            dim=-1,
        )
        target_tokens_atts = torch.cat([ans_mask_wo_bos, eos_col], dim=-1)

        # ---- 3) Get embeddings for instructions and answers ----
        embed_instr = embed_tokens(self.instruc_tokens["input_ids"])
        ins1 = embed_instr[0][self.instruc_tokens["attention_mask"][0].bool()]
        ins2 = embed_instr[1][self.instruc_tokens["attention_mask"][1].bool()][1:]
        ans_emb = embed_tokens(ans_ids_wo_bos)

        # ---- 4) Build full prompt embeddings ----
        embeddings, target_lens = [], []
        for seq_emb, a_emb, a_att in zip(seq_embeddings, ans_emb, ans_mask_wo_bos):
            full = torch.cat([ins1, seq_emb, ins2, a_emb[a_att]], dim=0)
            target_lens.append(a_att.sum())
            embeddings.append(full)

        inputs_embeds, attention_mask = stack_and_pad_left(embeddings)
        inputs_embeds, attention_mask = (
            inputs_embeds.to(device),
            attention_mask.to(device),
        )

        # Label mask for loss computation
        label_mask = attention_mask.clone()
        for i in range(label_mask.shape[0]):
            label_mask[i, : -target_lens[i] - 1] = 0
        label_mask = label_mask.bool()

        if memory_debug >= 2 and inputs_embeds.shape[1] > 4096:
            print("inputs_embeds.shape:", inputs_embeds.shape)  # (B, T, H)

        logits = self.Llama_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask
        ).logits
        return logits[label_mask], target_tokens_ids[target_tokens_atts]

    def forward(self, inputs, seq_positions):
        """
        :param inputs: tokenized log lines for Llama (concatenated across sequences), dict with input_ids [N_lines, T]
        :param seq_positions: list[int] start positions (excluding initial 0) to split lines back into sequences
        :return: generated token ids, shape [B, gen_len]
        """
        device = self.device
        embed_tokens = self._get_embed_tokens()

        # Per-line mean-pooled embeddings
        seq_embeddings = self._llama_embed_tokens(
            inputs, seq_positions, pooling=self.embed_pooling
        )
        batch_size = len(seq_embeddings)

        # Instruction embeddings
        embed_instr = embed_tokens(self.instruc_tokens["input_ids"])  # [2, L, H]
        ins1 = embed_instr[0][self.instruc_tokens["attention_mask"][0].bool()]
        ins2 = embed_instr[1][self.instruc_tokens["attention_mask"][1].bool()][1:]

        # Answer prefix "The sequence is"
        prefix = "The sequence is"
        prefix_ids = self.Llama_tokenizer(prefix, return_tensors="pt")["input_ids"][
            0, 1:
        ].to(device)
        prefix_emb = embed_tokens(prefix_ids)  # [P, H]

        # Build prompts
        prompts = [
            torch.cat([ins1, seq_emb, ins2, prefix_emb], dim=0)
            for seq_emb in seq_embeddings
        ]
        inputs_embeds, attention_mask = stack_and_pad_left(prompts)
        inputs_embeds = inputs_embeds.to(device)
        attention_mask = attention_mask.to(device)

        # Generation loop (greedy), capped to short length like original
        eos_id = self.Llama_tokenizer.eos_token_id
        pad_id = self.Llama_tokenizer.pad_token_id
        eos = torch.tensor([eos_id], device=device)
        unfinished = torch.ones(batch_size, dtype=torch.long, device=device)

        cache = DynamicCache()
        answers, finished = [], False
        next_emb = None

        while not finished:
            if len(cache) == 0:
                out = self.Llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                )
            else:
                out = self.Llama_model(
                    inputs_embeds=next_emb[:, None, :],
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                )

            logits = out.logits
            next_ids = torch.argmax(logits[:, -1, :], dim=-1)
            next_ids = next_ids * unfinished + pad_id * (1 - unfinished)
            answers.append(next_ids)

            next_emb = embed_tokens(next_ids)
            attention_mask = torch.cat([attention_mask, unfinished[:, None]], dim=1)

            # stop on EOS or after 5 tokens (to match your short answers)
            unfinished = unfinished.mul(next_ids.ne(eos).prod(dim=0))
            if unfinished.max() == 0 or len(answers) > 5:
                finished = True

        return torch.stack(answers, dim=1)  # [B, gen_len]

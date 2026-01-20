import os.path

import peft
import torch
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from torch import nn
from transformers import (
    BertTokenizerFast,
    BertModel,
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


def merge_data(data):
    merged_data = []

    # 用于记录每个子列表开始的位置
    start_positions = []

    # 当前起始位置
    current_position = 0

    for sublist in data:
        start_positions.append(current_position)
        merged_data.extend(sublist)
        current_position += len(sublist)
    return merged_data, start_positions


def stack_and_pad_right(tensors):
    # 找到第一维度的最大长度
    max_len = max(tensor.shape[0] for tensor in tensors)

    # 创建一个存放结果的列表
    padded_tensors = []
    padding_masks = []

    for tensor in tensors:
        # 计算需要填充的长度
        pad_len = max_len - tensor.shape[0]

        # 使用零填充
        padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_len))
        padded_tensors.append(padded_tensor)

        # 创建填充位置的掩码
        padding_mask = torch.cat(
            [
                torch.ones(tensor.shape[0], dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ]
        )
        padding_masks.append(padding_mask)

    # 堆叠所有填充后的张量
    stacked_tensor = torch.stack(padded_tensors)
    padding_masks = torch.stack(padding_masks)

    return stacked_tensor, padding_masks


def stack_and_pad_left(tensors):
    # 找到第一维度的最大长度
    max_len = max(tensor.shape[0] for tensor in tensors)

    # 创建一个存放结果的列表
    padded_tensors = []
    padding_masks = []

    for tensor in tensors:
        # 计算需要填充的长度
        pad_len = max_len - tensor.shape[0]

        # 使用零填充
        padded_tensor = torch.nn.functional.pad(tensor, (0, 0, pad_len, 0))
        padded_tensors.append(padded_tensor)

        # 创建填充位置的掩码
        padding_mask = torch.cat(
            [
                torch.zeros(pad_len, dtype=torch.long),
                torch.ones(tensor.shape[0], dtype=torch.long),
            ]
        )
        padding_masks.append(padding_mask)

    # 堆叠所有填充后的张量
    stacked_tensor = torch.stack(padded_tensors)
    padding_masks = torch.stack(padding_masks)

    return stacked_tensor, padding_masks


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # load the model into memory using 4-bit precision
    bnb_4bit_use_double_quant=False,  # use double quantition
    bnb_4bit_quant_type="nf4",  # use NormalFloat quantition
    bnb_4bit_compute_dtype=torch.bfloat16,  # use hf for computing when we need
)


class LogLLM(nn.Module):
    def __init__(
        self,
        Bert_path,
        Llama_path,
        ft_path=None,
        is_train_mode=True,
        device=torch.device("cuda:0"),
        max_content_len=128,
        max_seq_len=128,
    ):
        super().__init__()
        self.max_content_len = (
            max_content_len  # max length of each log messages (contents)
        )
        self.max_seq_len = max_seq_len  # max length of each log sequence  (log sequence contains some log messages)
        self.device = device
        self.Llama_tokenizer = AutoTokenizer.from_pretrained(
            Llama_path, padding_side="right"
        )
        self.Llama_tokenizer.pad_token = self.Llama_tokenizer.eos_token

        self.Llama_model = AutoModelForSequenceClassification.from_pretrained(
            Llama_path,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            device_map=device,
            attn_implementation="flash_attention_2",
        )  # embedding dim = 4096
        self.Llama_model.config.num_labels = 2  # normal or anomalous
        # fixes: Cannot handle batch sizes > 1 if no padding token is defined.
        self.Llama_model.config.pad_token_id = self.Llama_tokenizer.pad_token_id
        # turn off the KV‑cache for multi‑sample inference
        self.Llama_model.config.use_cache = False

        self.Bert_tokenizer = BertTokenizerFast.from_pretrained(
            Bert_path, do_lower_case=True
        )
        self.Bert_model = BertModel.from_pretrained(
            Bert_path,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            device_map=device,
        )

        self.projector = nn.Linear(
            self.Bert_model.config.hidden_size,
            self.Llama_model.config.hidden_size,
            device=device,
        )
        # self.projector = nn.Linear(self.Bert_model.config.hidden_size, self.Llama_model.config.hidden_size).half().to(device)

        self.instruc_tokens = self.Llama_tokenizer(
            [
                "Below is a sequence of system log messages:",
                ". Is this sequence normal or anomalous? \\n",
            ],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # if is_train_mode:
        #     self.Bert_model = prepare_model_for_kbit_training(self.Bert_model)
        #     self.Llama_model = prepare_model_for_kbit_training(self.Llama_model)

        if ft_path is not None:
            print(f"Loading peft model from {ft_path}.")
            Llama_ft_path = os.path.join(ft_path, "Llama_ft")
            Bert_ft_path = os.path.join(ft_path, "Bert_ft")
            projector_path = os.path.join(ft_path, "projector.pt")
            self.Llama_model = PeftModel.from_pretrained(
                self.Llama_model,
                Llama_ft_path,
                is_trainable=is_train_mode,
                torch_dtype=torch.float16,
            )
            self.Bert_model = PeftModel.from_pretrained(
                self.Bert_model,
                Bert_ft_path,
                is_trainable=is_train_mode,
                torch_dtype=torch.float16,
            )
            self.projector.load_state_dict(
                torch.load(projector_path, map_location=device, weights_only=True)
            )
        else:
            print("Creating peft model.")
            Bert_peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=4,
                lora_alpha=32,
                lora_dropout=0.01,
            )
            self.Bert_model = get_peft_model(self.Bert_model, Bert_peft_config)

            Llama_peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type=TaskType.SEQ_CLS,
            )
            self.Llama_model = get_peft_model(self.Llama_model, Llama_peft_config)

            for n, p in self.Llama_model.named_parameters():
                if "lora" in n.lower():
                    # move the *weight* to FP32 (the optimizer will see FP32 grads)
                    p.data = p.data.to(torch.float32)
            # The PEFT wrapper creates a `modules_to_save` dict that holds the
            # original (un‑adapted) head.  Cast it to FP32 as well.
            if hasattr(self.Llama_model.base_model.model, "score"):
                score = self.Llama_model.base_model.model.score
                for n, p in score.named_parameters():
                    p.data = p.data.to(torch.float32)

    def save_ft_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        Llama_ft_path = os.path.join(path, "Llama_ft")
        Bert_ft_path = os.path.join(path, "Bert_ft")
        projector_path = os.path.join(path, "projector.pt")
        self.Llama_model.save_pretrained(Llama_ft_path, safe_serialization=True)
        self.Bert_model.save_pretrained(Bert_ft_path, safe_serialization=True)
        torch.save(self.projector.state_dict(), projector_path)

    def set_train_only_projector(self):
        for name, param in self.projector.named_parameters():
            param.requires_grad = True
        for name, param in self.Bert_model.named_parameters():
            param.requires_grad = False
        for name, param in self.Llama_model.named_parameters():
            param.requires_grad = False
        for name, param in self.Llama_model.score.named_parameters():
            param.requires_grad = False

    def set_train_only_Llama(self):
        for name, param in self.projector.named_parameters():
            param.requires_grad = False
        for name, param in self.Bert_model.named_parameters():
            param.requires_grad = False
        for name, param in self.Llama_model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        for name, param in self.Llama_model.score.named_parameters():
            param.requires_grad = True

    def set_train_projectorAndBert(self):
        for name, param in self.projector.named_parameters():
            param.requires_grad = True
        for name, param in self.Bert_model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        for name, param in self.Llama_model.named_parameters():
            param.requires_grad = False
        for name, param in self.Llama_model.score.named_parameters():
            param.requires_grad = False

    def set_finetuning_all(self):
        for name, param in self.projector.named_parameters():
            param.requires_grad = True
        for name, param in self.Bert_model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        for name, param in self.Llama_model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        for name, param in self.Llama_model.score.named_parameters():
            param.requires_grad = True

    def train_helper(self, inputs, seq_positions, labels):
        """
        :param inputs: tokenized sequences for BERT (concatenated)
        :param seq_positions: boundaries for splitting BERT outputs
        :param labels: np.array of labels, values in ['anomalous', 'normal']
        :return: (logits, labels_tensor)
        """

        # Convert string labels → integers
        # anomalous -> 1, normal -> 0
        label_map = {"normal": 0, "anomalous": 1}
        labels_tensor = torch.tensor(
            [label_map[l] for l in labels],
            device=self.device,
            dtype=torch.long,
        )

        # === BERT ENCODING ===
        outputs = self.Bert_model(**inputs).pooler_output
        outputs = self.projector(outputs.float()).half()

        # Split into per-sequence embeddings
        seq_embeddings = torch.tensor_split(outputs, seq_positions)

        # === BUILD INSTRUCTION + SEQ PROMPT EMBEDDINGS ===
        if isinstance(self.Llama_model, peft.peft_model.PeftModelForSequenceClassification):
            instruc_embeddings = self.Llama_model.model.model.embed_tokens(
                self.instruc_tokens["input_ids"]
            )
        else:
            instruc_embeddings = self.Llama_model.model.embed_tokens(
                self.instruc_tokens["input_ids"]
            )

        ins1 = instruc_embeddings[0][self.instruc_tokens["attention_mask"][0].bool()]
        ins2 = instruc_embeddings[1][self.instruc_tokens["attention_mask"][1].bool()][1:]

        # Build prompt embeddings for each sequence
        embeddings = []
        for seq_embedding in seq_embeddings:
            full_prompt = torch.cat([ins1, seq_embedding, ins2])
            embeddings.append(full_prompt)

        # Pad to batch format
        inputs_embeds, attention_mask = stack_and_pad_left(embeddings)
        inputs_embeds = inputs_embeds.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # === CLASSIFICATION FORWARD ===
        logits = self.Llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        ).logits  # shape: [batch, 2]

        return logits, labels_tensor

    def forward(self, inputs, seq_positions):
        # BERT encodings
        outputs = self.Bert_model(**inputs).pooler_output
        outputs = outputs.float()
        outputs = self.projector(outputs).half()

        seq_embeddings = torch.tensor_split(outputs, seq_positions)

        # Build prompt embeddings (same as in train-helper)
        if isinstance(self.Llama_model, peft.peft_model.PeftModelForSequenceClassification):
            instruc_embeddings = self.Llama_model.model.model.embed_tokens(self.instruc_tokens['input_ids'])
        else:
            instruc_embeddings = self.Llama_model.model.embed_tokens(self.instruc_tokens['input_ids'])

        ins1 = instruc_embeddings[0][self.instruc_tokens['attention_mask'][0].bool()]
        ins2 = instruc_embeddings[1][self.instruc_tokens['attention_mask'][1].bool()][1:]

        prompt_embeddings = []
        for seq_embedding in seq_embeddings:
            emb = torch.cat([ins1, seq_embedding, ins2])
            prompt_embeddings.append(emb)

        # Pad and combine
        inputs_embeds, attention_mask = stack_and_pad_left(prompt_embeddings)
        inputs_embeds = inputs_embeds.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # LLaMA classification
        outputs = self.Llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )

        # outputs.logits: [batch, 2]
        return outputs.logits

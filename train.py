import os
import sys
from functools import partial
from pathlib import Path
import torch
from torch.amp import GradScaler
from tqdm import tqdm
from torch import nn, autocast
from torch.utils.data import DataLoader
from customDataset import CustomDataset, CustomCollator, BalancedSampler
from torch import optim
import eval as meval

dataset_name = os.getenv("DATASET", "HDFS")  # 'Thunderbird' 'HDFS_v1'  'BGL' 'Liberty'
base = os.getenv("BASE")
memory_debug = int(os.getenv("MEMORY_DEBUG", "0"))
run_validation = bool(int(os.getenv("RUN_VALIDATION", "0")))

n_epochs_1 = 1
n_epochs_2_1 = 1
n_epochs_2_2 = 1
n_epochs_3 = int(os.getenv("N_EPOCHS_3", "2"))
batch_size = 16
micro_batch_size = int(os.getenv("MICRO_BATCH_SIZE", "4"))
gradient_accumulation_steps = batch_size // micro_batch_size


lr_1 = 5e-4
lr_2_1 = 5e-4
lr_2_2 = 5e-5
lr_3 = 5e-5
max_content_len = 100
max_seq_len = 128

data_path = f"{base}/data/{dataset_name}/train.csv"

min_less_portion = 0.3

Bert_path = f"{base}/bert-base-uncased"
Llama_path = f"{base}/Meta-Llama-3-8B"

ROOT_DIR = Path(__file__).parent
ft_prefix = os.getenv("FT_PREFIX")
if ft_prefix:
    ft_path = os.path.join(ROOT_DIR, ft_prefix, f"ft_model_{dataset_name}")
else:
    ft_path = os.path.join(ROOT_DIR, r"ft_model_{}".format(dataset_name))

device = torch.device("cuda:0")

print(
    f"n_epochs_1: {n_epochs_1}\n"
    f"n_epochs_2_1: {n_epochs_2_1}\n"
    f"n_epochs_2_2: {n_epochs_2_2}\n"
    f"n_epochs_3: {n_epochs_3}\n"
    f"dataset_name: {dataset_name}\n"
    f"batch_size: {batch_size}\n"
    f"micro_batch_size: {micro_batch_size}\n"
    f"lr_1: {lr_1}\n"
    f"lr_2_1: {lr_2_1}\n"
    f"lr_2_2: {lr_2_2}\n"
    f"lr_3: {lr_3}\n"
    f"max_content_len: {max_content_len}\n"
    f"max_seq_len: {max_seq_len}\n"
    f"min_less_portion: {min_less_portion}\n"
    f"device: {device}"
)


def print_number_of_trainable_model_parameters(model):
    params = set()
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            params.add(param)
            trainable_model_params += param.numel()
    print(
        f"all params num: {all_model_params}, trainable param num: {trainable_model_params}"
    )
    return params


def trained_parameters(model):
    yield from (p for p in model.parameters() if p.requires_grad)


def trainModel(model, dataloader, gradient_accumulation_steps, n_epochs, lr, val_callback=None):
    criterion = nn.CrossEntropyLoss(reduction="mean")

    trainable_model_params = print_number_of_trainable_model_parameters(model)
    optimizer = torch.optim.AdamW(trainable_model_params, lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

    normal_tokens = model.Llama_tokenizer("The sequence is normal.")["input_ids"]
    anomalous_tokens = model.Llama_tokenizer("The sequence is anomalous.")["input_ids"]
    special_normal_tokens = set(normal_tokens) - set(anomalous_tokens)
    special_anomalous_tokens = set(anomalous_tokens) - set(normal_tokens)

    total_steps = n_epochs * len(dataloader)
    scheduler_step = max(int(total_steps / 10), 1)

    print(f"scheduler_step: {scheduler_step}")

    steps = 0
    for epoch in range(int(n_epochs)):
        total_acc, total_acc_count, total_count, train_loss = 0, 0, 0, 0

        pbar = tqdm(dataloader, desc="Epoch {}/{}".format(epoch, n_epochs))
        for i_th, bathc_i in enumerate(pbar):
            steps += 1

            inputs = bathc_i["inputs"]
            seq_positions = bathc_i["seq_positions"]
            labels = bathc_i["labels"]

            inputs = inputs.to(device)
            seq_positions = seq_positions

            outputs, targets = model.train_helper(inputs, seq_positions, labels)

            loss = criterion(outputs, targets)
            loss = loss / gradient_accumulation_steps

            loss.backward()
            # print(loss)

            if ((i_th + 1) % gradient_accumulation_steps == 0) or (
                (i_th + 1) == len(dataloader)
            ):
                # optimizer the net
                optimizer.step()  # 更新网络参数
                optimizer.zero_grad()  # reset grdient # 清空过往梯度

            acc_mask = torch.zeros_like(targets, device=device).bool()
            for token in special_normal_tokens.union(special_anomalous_tokens):
                acc_mask[targets == token] = True

            total_acc += (outputs.argmax(1)[acc_mask] == targets[acc_mask]).sum().item()
            total_acc_count += acc_mask.sum()

            train_loss += loss.item() * gradient_accumulation_steps * targets.size(0)

            total_count += targets.size(0)

            if steps % scheduler_step == 0:
                scheduler.step()
            pbar.set_postfix(
                lr=scheduler.get_last_lr()[0],
                loss=loss.item() * gradient_accumulation_steps,
            )

            # ---- MEMORY DEBUG SECTION ----
            if memory_debug >= 2 and steps % 100 == 0:
                torch.cuda.synchronize()
                print(
                    f"\n[Step {steps}] GPU Memory Summary:\n"
                    f"{torch.cuda.memory_summary(device=device, abbreviated=True)}\n"
                )
            # --------------------------------

            if steps % 10000 == 0:  # every 10000 steps, print loss and acc
                train_loss_epoch = train_loss / total_count
                train_acc_epoch = total_acc / total_acc_count
                print(
                    f"[Epoch {epoch + 1:{len(str(n_epochs))}}/{n_epochs}] "
                    f"[loss: {train_loss_epoch:3f}]"
                    f"[acc: {train_acc_epoch:3f}]"
                )

                total_acc, total_acc_count, total_count, train_loss = 0, 0, 0, 0

        if total_count > 0:
            train_loss_epoch = train_loss / total_count
            train_acc_epoch = total_acc / total_acc_count
            print(
                f"[Epoch {epoch + 1:{len(str(n_epochs))}}/{n_epochs}] "
                f"[loss: {train_loss_epoch:3f}]"
                f"[acc: {train_acc_epoch:3f}]"
            )

        if val_callback is not None:
            model.eval()
            with torch.no_grad():
                val_callback(model)
            model.train()

    if memory_debug >= 1:
        torch.cuda.synchronize()
        print(
            f"\n[Training Completed] GPU Memory Summary:\n"
            f"{torch.cuda.memory_summary(device=device, abbreviated=True)}\n"
        )



def trainModelAMP(model, dataloader, gradient_accumulation_steps, n_epochs, lr, val_callback=None):
    criterion = nn.CrossEntropyLoss(reduction="mean")

    trainable_model_params = print_number_of_trainable_model_parameters(model)
    optimizer = torch.optim.AdamW(trainable_model_params, lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
    scaler = GradScaler()

    total_steps = n_epochs * len(dataloader)
    scheduler_step = max(int(total_steps / 10), 1)

    print(f"scheduler_step: {scheduler_step}")

    steps = 0
    for epoch in range(int(n_epochs)):
        total_count, train_loss = 0, 0
        ttl_norm = 0.0

        pbar = tqdm(dataloader, desc="Epoch {}/{}".format(epoch, n_epochs))
        for i_th, bathc_i in enumerate(pbar):
            steps += 1

            inputs = bathc_i["inputs"]
            seq_positions = bathc_i["seq_positions"]
            labels = bathc_i["labels"]

            inputs = inputs.to(device)
            seq_positions = seq_positions
            with autocast(device_type='cuda'):
                outputs, targets = model.train_helper(inputs, seq_positions, labels)
                loss = criterion(outputs, targets)
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if ((i_th + 1) % gradient_accumulation_steps == 0) or (
                (i_th + 1) == len(dataloader)
            ):
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                ttl_norm = nn.utils.clip_grad_norm_(trained_parameters(model), 1.0).detach().item()

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * gradient_accumulation_steps * targets.size(0)

            total_count += targets.size(0)

            if steps % scheduler_step == 0:
                scheduler.step()

            pbar.set_postfix(
                lr=scheduler.get_last_lr()[0],
                loss=loss.item() * gradient_accumulation_steps,
                grad_norm=ttl_norm,
            )

            # ---- MEMORY DEBUG SECTION ----
            if memory_debug >= 2 and steps % 100 == 0:
                torch.cuda.synchronize()
                print(
                    f"\n[Step {steps}] GPU Memory Summary:\n"
                    f"{torch.cuda.memory_summary(device=device, abbreviated=True)}\n"
                )
            # --------------------------------

            if steps % 10000 == 0:  # every 10000 steps, print loss and acc
                train_loss_epoch = train_loss / total_count
                print(
                    f"[Epoch {epoch + 1:{len(str(n_epochs))}}/{n_epochs}] "
                    f"[loss: {train_loss_epoch:3f}]"
                )

                total_acc, total_acc_count, total_count, train_loss = 0, 0, 0, 0

        if total_count > 0:
            train_loss_epoch = train_loss / total_count
            print(
                f"[Epoch {epoch + 1:{len(str(n_epochs))}}/{n_epochs}] "
                f"[loss: {train_loss_epoch:3f}]"
            )

        if val_callback is not None:
            model.eval()
            with torch.no_grad():
                val_callback(model)
            model.train()

    if memory_debug >= 1:
        torch.cuda.synchronize()
        print(
            f"\n[Training Completed] GPU Memory Summary:\n"
            f"{torch.cuda.memory_summary(device=device, abbreviated=True)}\n"
        )


if __name__ == "__main__":
    accepted_modes = ["nobert", "seqcls", "base", "onemore"]
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    elif ft_prefix in accepted_modes:
        # If not specified from command line, use FT_PREFIX as mode if valid
        mode = ft_prefix
    else:
        mode = "base"

    print(f"mode: {mode}")
    print(f"dataset: {data_path}")
    dataset = CustomDataset(data_path, drop_duplicates=False)
    eval_fn = meval.evalModel

    if mode in ("base", "onemore"):
        from model import LogLLM

        model = LogLLM(
            Bert_path,
            Llama_path,
            device=device,
            max_content_len=max_content_len,
            max_seq_len=max_seq_len,
        )

        tokenizer = model.Bert_tokenizer
    elif mode == "nobert":
        from model_nobert import LogLLM

        model = LogLLM(
            Llama_path,
            is_train_mode=True,
            device=device,
            max_content_len=max_content_len,
            max_seq_len=max_seq_len,
        )

        tokenizer = model.Llama_tokenizer
    elif mode == "seqcls":
        from model_seqcls import LogLLM

        model = LogLLM(
            Bert_path,
            Llama_path,
            device=device,
            max_content_len=max_content_len,
            max_seq_len=max_seq_len,
        )

        tokenizer = model.Bert_tokenizer

        trainModel = trainModelAMP
        eval_fn = meval.evalModelSeqCls
    else:
        raise ValueError(f"Unknown mode: {mode}")

    collator = CustomCollator(
        tokenizer, max_seq_len=max_seq_len, max_content_len=max_content_len
    )

    if run_validation:
        val_dataset = CustomDataset(meval.data_path)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=micro_batch_size*8,
            collate_fn=collator,
            num_workers=min(int(os.getenv("PBS_NCPUS", os.cpu_count())), 4),
            shuffle=False,
            drop_last=False,
        )
        val_callback = partial(eval_fn, dataloader=val_dataloader)
    else:
        val_callback = None

    # phase 1
    if mode != "onemore":
        dataloader_max_samples = DataLoader(
            dataset,
            batch_size=micro_batch_size,
            num_workers=min(int(os.getenv("PBS_NCPUS", os.cpu_count())), 4),
            sampler=BalancedSampler(
                dataset, target_ratio=min_less_portion, max_samples=1000
            ),
            collate_fn=collator,
            drop_last=True,
        )

        print("*" * 10 + "Start training Llama" + "*" * 10)
        model.set_train_only_Llama()
        trainModel(
            model, dataloader_max_samples, gradient_accumulation_steps, n_epochs_1, lr_1, val_callback=val_callback
        )
        model.save_ft_model(ft_path)
        del dataloader_max_samples

    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=min(int(os.getenv("PBS_NCPUS", os.cpu_count())), 4),
        sampler=BalancedSampler(dataset, target_ratio=min_less_portion),
        collate_fn=collator,
        drop_last=True,
    )

    if mode in ("base", "seqcls"):
        # phase 2-1
        print("*" * 10 + "Start training projector" + "*" * 10)
        model.set_train_only_projector()
        trainModel(model, dataloader, gradient_accumulation_steps, n_epochs_2_1, lr_2_1, val_callback=val_callback)
        model.save_ft_model(ft_path)

        # phase 2-2
        print("*" * 10 + "Start training projector and Bert" + "*" * 10)
        model.set_train_projectorAndBert()
        trainModel(model, dataloader, gradient_accumulation_steps, n_epochs_2_2, lr_2_2, val_callback=val_callback)
        model.save_ft_model(ft_path)

    model.set_finetuning_all()
    print("*" * 10 + "Start training entire model" + "*" * 10)
    trainModel(model, dataloader, gradient_accumulation_steps, n_epochs_3, lr_3, val_callback=val_callback)
    model.save_ft_model(ft_path)

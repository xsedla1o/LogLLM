import os
import re
import sys
from pathlib import Path
import numpy as np
import torch
from torch import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from customDataset import CustomDataset, CustomCollator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

max_content_len = 100
max_seq_len = 128
batch_size = int(os.getenv("BATCH_SIZE", "32"))
dataset_name = os.getenv("DATASET", "HDFS")  # 'Thunderbird' 'HDFS_v1'  'BGL'
memory_debug = int(os.getenv("MEMORY_DEBUG", "0"))

base = os.getenv("BASE")
data_path = f"{base}/data/{dataset_name}/test.csv"

Bert_path = f"{base}/bert-base-uncased"
Llama_path = f"{base}/Meta-Llama-3-8B"

ROOT_DIR = Path(__file__).parent
ft_prefix = os.getenv("FT_PREFIX")
if ft_prefix:
    ft_path = os.path.join(ROOT_DIR, ft_prefix, f"ft_model_{dataset_name}")
else:
    ft_path = os.path.join(ROOT_DIR, r"ft_model_{}".format(dataset_name))

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(
    f"dataset_name: {dataset_name}\n"
    f"batch_size: {batch_size}\n"
    f"max_content_len: {max_content_len}\n"
    f"max_seq_len: {max_seq_len}\n"
    f"device: {device}"
)


def evalModel(model, dataloader):
    model.eval()

    preds = []

    with torch.no_grad():
        for bathc_i in tqdm(dataloader):
            inputs = bathc_i["inputs"]
            seq_positions = bathc_i["seq_positions"]

            inputs = inputs.to(device)
            seq_positions = seq_positions

            outputs_ids = model(inputs, seq_positions)
            outputs = model.Llama_tokenizer.batch_decode(outputs_ids)

            # print(outputs)

            for text in outputs:
                match = re.search(r"normal|anomalous", text, re.IGNORECASE)
                if match:
                    preds.append(match.group())
                else:
                    print(f"error :{text}")
                    preds.append("")

    preds_copy = np.array(preds)
    preds = np.zeros_like(preds_copy, dtype=int)
    preds[preds_copy == "anomalous"] = 1
    preds[preds_copy != "anomalous"] = 0
    gt = dataloader.dataset.get_label()

    precision = precision_score(gt, preds, average="binary", pos_label=1)
    recall = recall_score(gt, preds, average="binary", pos_label=1)
    f = f1_score(gt, preds, average="binary", pos_label=1)
    acc = accuracy_score(gt, preds)

    num_anomalous = (gt == 1).sum()
    num_normal = (gt == 0).sum()

    print(
        f"Number of anomalous seqs: {num_anomalous}; number of normal seqs: {num_normal}"
    )

    pred_num_anomalous = (preds == 1).sum()
    pred_num_normal = (preds == 0).sum()

    print(
        f"Number of detected anomalous seqs: {pred_num_anomalous}; number of detected normal seqs: {pred_num_normal}"
    )

    print(f"precision: {precision}, recall: {recall}, f1: {f}, acc: {acc}")


def evalModelSeqCls(model, dataloader):
    model.eval()

    preds = []

    with torch.no_grad():
        for batch_i in tqdm(dataloader):
            inputs = batch_i["inputs"].to(device)
            seq_positions = batch_i["seq_positions"]

            # model now returns logits: [batch, 2]
            with autocast(device_type='cuda'):
                logits = model(inputs, seq_positions)

            # predicted class: 0 or 1
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()

            preds.extend(batch_preds)

    preds = np.array(preds, dtype=int)

    # ground truth labels must be 0/1 integers
    gt = dataloader.dataset.get_label()

    precision = precision_score(gt, preds, average="binary", pos_label=1)
    recall = recall_score(gt, preds, average="binary", pos_label=1)
    f = f1_score(gt, preds, average="binary", pos_label=1)
    acc = accuracy_score(gt, preds)

    num_anomalous = (gt == 1).sum()
    num_normal = (gt == 0).sum()

    print(f"Number of anomalous seqs: {num_anomalous}; number of normal seqs: {num_normal}")

    pred_num_anomalous = (preds == 1).sum()
    pred_num_normal = (preds == 0).sum()

    print(f"Number of detected anomalous seqs: {pred_num_anomalous}; number of detected normal seqs: {pred_num_normal}")

    print(f"precision: {precision}, recall: {recall}, f1: {f}, acc: {acc}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "base"

    print(f"dataset: {data_path}")
    dataset = CustomDataset(data_path)

    if mode in ("base", "onemore"):
        from model import LogLLM

        model = LogLLM(
            Bert_path,
            Llama_path,
            ft_path=ft_path,
            is_train_mode=False,
            device=device,
            max_content_len=max_content_len,
            max_seq_len=max_seq_len,
        )

        tokenizer = model.Bert_tokenizer
    elif mode == "nobert":
        from model_nobert import LogLLM

        model = LogLLM(
            Llama_path,
            ft_path=ft_path,
            is_train_mode=False,
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
            ft_path=ft_path,
            device=device,
            max_content_len=max_content_len,
            max_seq_len=max_seq_len,
        )

        tokenizer = model.Bert_tokenizer
        evalModel = evalModelSeqCls
    else:
        raise ValueError(f"Unknown mode: {mode}")

    collator = CustomCollator(
        tokenizer, max_seq_len=max_seq_len, max_content_len=max_content_len
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=min(int(os.getenv("PBS_NCPUS", os.cpu_count())), 4),
        shuffle=False,
        drop_last=False,
    )

    evalModel(model, dataloader)
    if memory_debug >= 1 and torch.cuda.is_available():
        print(f"Memory Summary Before Emptying Cache:\n")
        device = torch.device("cuda:0")
        torch.cuda.synchronize()
        print(
            f"{torch.cuda.memory_summary(device=device, abbreviated=True)}\n"
        )

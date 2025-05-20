# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import random
import math
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad
from utils import *
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank","--local-rank", type=int, default=-1)
parser.add_argument("--bin_num", type=int, default=5)
parser.add_argument("--gene_num", type=int, default=16906)
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--batch_size", type=int, default=3)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--grad_acc", type=int, default=60)
parser.add_argument("--valid_every", type=int, default=1)
parser.add_argument("--pos_embed", type=bool, default=True)
parser.add_argument("--data_path", type=str, default='./data/Zheng68K.h5ad')
parser.add_argument("--model_path", type=str, default='./panglao_pretrained.pth')
parser.add_argument("--ckpt_dir", type=str, default='./ckpts/')
parser.add_argument("--model_name", type=str, default='finetune')
args = parser.parse_args()

rank = int(os.environ["RANK"])
local_rank = args.local_rank
is_master = local_rank == 0

SEED = args.seed
BATCH_SIZE = args.batch_size
SEQ_LEN = args.gene_num + 1
POS_EMBED_USING = args.pos_embed
CLASS = args.bin_num + 2

seed_all(SEED + torch.distributed.get_rank())
dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
world_size = torch.distributed.get_world_size()

class SCDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        seq_label = self.label[rand_start]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]

data = sc.read_h5ad(args.data_path)
label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)
with open('label_dict', 'wb') as fp:
    pkl.dump(label_dict, fp)
with open('label', 'wb') as fp:
    pkl.dump(label, fp)
label = torch.from_numpy(label)
data = data.X

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
for index_train, index_val in sss.split(data, label):
    data_train, label_train = data[index_train], label[index_train]
    data_val, label_val = data[index_val], label[index_val]
    train_dataset = SCDataset(data_train, label_train)
    val_dataset = SCDataset(data_val, label_val)

train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

model = PerformerLM(
    num_tokens=CLASS,
    dim=200,
    depth=6,
    max_seq_len=SEQ_LEN,
    heads=10,
    local_attn_heads=0,
    g2v_position_emb=POS_EMBED_USING
)

ckpt = torch.load(args.model_path)
model.load_state_dict(ckpt['model_state_dict'])
model.to_out = nn.Identity()
model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

loss_fn = nn.CrossEntropyLoss().to(local_rank)
UNASSIGN_THRES = 0.0

model.eval()
dist.barrier()
with torch.no_grad():
    running_loss = 0.0
    cum_acc = 0.0
    for index, (data, labels) in enumerate(train_loader):
        index += 1
        data, labels = data.to(device), labels.to(device)
        logits = model(data)
        loss = loss_fn(logits, labels)
        running_loss += loss.item()

        softmax = nn.Softmax(dim=-1)
        final = softmax(logits).argmax(dim=-1)
        pred_num = labels.size(0)
        correct_num = torch.eq(final, labels).sum(dim=-1)
        cum_acc += torch.true_divide(correct_num, pred_num).mean().item()

    epoch_loss = running_loss / index
    epoch_acc = 100 * cum_acc / index
    epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
    epoch_acc = get_reduced(epoch_acc, local_rank, 0, world_size)
    if is_master:
        print(f'== Inference | Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}% ==')

predictions = []
truths = []
running_loss = 0.0
model.eval()
dist.barrier()
with torch.no_grad():
    for index, (data_v, labels_v) in enumerate(val_loader):
        index += 1
        data_v, labels_v = data_v.to(device), labels_v.to(device)
        logits = model(data_v)
        loss = loss_fn(logits, labels_v)
        running_loss += loss.item()

        softmax = nn.Softmax(dim=-1)
        final_prob = softmax(logits)
        final = final_prob.argmax(dim=-1)
        final[np.amax(np.array(final_prob.cpu()), axis=-1) < UNASSIGN_THRES] = -1
        predictions.append(final)
        truths.append(labels_v)

    predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_sampler.dataset), world_size)
    truths = distributed_concat(torch.cat(truths, dim=0), len(val_sampler.dataset), world_size)
    no_drop = predictions != -1
    predictions = np.array((predictions[no_drop]).cpu())
    truths = np.array((truths[no_drop]).cpu())
    cur_acc = accuracy_score(truths, predictions)
    f1 = f1_score(truths, predictions, average='macro')
    val_loss = running_loss / index
    val_loss = get_reduced(val_loss, local_rank, 0, world_size)

    if is_master:
        print(f'== Validation | Loss: {val_loss:.6f} | F1 Score: {f1:.6f} ==')
        print(confusion_matrix(truths, predictions))
        print(classification_report(truths, predictions, target_names=label_dict.tolist(), digits=4))

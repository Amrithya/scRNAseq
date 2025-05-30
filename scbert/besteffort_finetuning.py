# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import random
import math
import signal
import sys
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import torch
import umap.umap_ as umap
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad
from utils import *
import pickle as pkl
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
parser.add_argument("--bin_num", type=int, default=5)
parser.add_argument("--gene_num", type=int, default=16906)
parser.add_argument("--epoch", type=int, default=30)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--grad_acc", type=int, default=60)
parser.add_argument("--valid_every", type=int, default=1)
parser.add_argument("--pos_embed", type=bool, default=True)
parser.add_argument("--data_path", type=str, default='./data/Zheng68K.h5ad')
parser.add_argument("--model_path", type=str, default='./panglao_pretrained.pth')
parser.add_argument("--ckpt_dir", type=str, default='./ckpts/')
parser.add_argument("--model_name", type=str, default='finetune')
parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")

args = parser.parse_args()
rank = int(os.environ["RANK"])
local_rank = args.local_rank
is_master = local_rank == 0

SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num + 1
VALIDATE_EVERY = args.valid_every
PATIENCE = 10
UNASSIGN_THRES = 0.0

CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed
model_name = args.model_name
ckpt_dir = args.ckpt_dir

dist.init_process_group(backend='nccl')
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
world_size = torch.distributed.get_world_size()

seed_all(SEED + torch.distributed.get_rank())
print(f"[Init] Using {world_size} GPUs, local_rank: {local_rank}")

def save_ckpt(epoch, model, optimizer, scheduler, val_loss, model_name, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_path = os.path.join(ckpt_dir, f"{model_name}_latest.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss
    }, latest_path)

def signal_handler(signum, frame):
    if is_master:
        print("[SIGNAL] Received signal. Saving checkpoint...")
        save_ckpt(start_epoch, model, optimizer, scheduler, 0.0, model_name, ckpt_dir)
    sys.exit(0)

signal.signal(signal.SIGUSR1, signal_handler)


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

class Identity(nn.Module):
    def __init__(self, dropout=0., h_dim=100, out_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(SEQ_LEN, out_dim)
        self.act1 = nn.ReLU()
        self._printed = False
        self.current_conv_output = None

    def forward(self, x):
        if not self._printed:
            print(f"Shape after Performer, before Identity: {x.shape}")
        x = x[:, None, :, :]  
        conv_out = self.conv1(x)
        self.current_conv_output = conv_out.view(conv_out.size(0), -1).detach().cpu()
        x = self.act(conv_out)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.act1(x)
        if not self._printed:
            print(f"Shape after Identity: {x.shape}")
            self._printed = True
        return x


try:
    print("Loading data...")
    data = sc.read_h5ad(args.data_path)
    label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)
    with open('label_dict', 'wb') as fp:
        pkl.dump(label_dict, fp)
    with open('label', 'wb') as fp:
        pkl.dump(label, fp)
    class_num = np.unique(label, return_counts=True)[1].tolist()
    class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])
    label = torch.from_numpy(label)
    data = data.X
except Exception as e:
    print(f"[ERROR] Data loading failed: {e}")
    exit(1)

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

print("Loading pretrained PerformerLM model...")
ckpt = torch.load(args.model_path, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])

for param in model.parameters():
    param.requires_grad = False
for param in model.norm.parameters():
    param.requires_grad = True
for param in model.performer.net.layers[-2].parameters():
    param.requires_grad = True

model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])
model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=15,
    cycle_mult=2,
    max_lr=LEARNING_RATE,
    min_lr=1e-6,
    warmup_steps=5,
    gamma=0.9
)
loss_fn = nn.CrossEntropyLoss(weight=None).to(local_rank)


start_epoch = 1
latest_ckpt_path = os.path.join(ckpt_dir, f"{model_name}_latest.pth")
if args.resume and os.path.exists(latest_ckpt_path):
    print(f"[RESUME] Loading checkpoint from {latest_ckpt_path}")
    checkpoint = torch.load(latest_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

dist.barrier()
trigger_times = 0
max_acc = 0.0

all_conv_features = []
all_labels_for_conv = []

for i in range(start_epoch, EPOCHS + 1):
    
    train_loader.sampler.set_epoch(i)
    model.train()
    dist.barrier()
    running_loss = 0.0
    cum_acc = 0.0

    try:
        for index, (data, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {i} - Training")):
            index += 1
            data, labels = data.to(device), labels.to(device)
            if index % GRADIENT_ACCUMULATION != 0:
                with model.no_sync():
                    logits = model(data)
                    loss = loss_fn(logits, labels)
                    loss.backward()
            if index % GRADIENT_ACCUMULATION == 0:
                logits = model(data)
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                optimizer.step()
                optimizer.zero_grad()
                
            running_loss += loss.item()
            final = torch.softmax(logits, dim=-1).argmax(dim=-1)
            cum_acc += (final == labels).float().mean().item()     
        
        print(f"Epoch {i}: Loss {running_loss / len(train_loader)}, Accuracy {cum_acc / len(train_loader)}")

    except Exception as e:
        print(f"[ERROR] Training loop interrupted: {e}")
        save_ckpt(i, model, optimizer, scheduler, running_loss / len(train_loader), model_name, ckpt_dir)
        sys.exit(0)

    model.eval()
    val_acc = 0.0
    val_loss = 0.0
    all_val_pred = []
    all_val_label = []

    with torch.no_grad():
        for data_v, label_v in tqdm(val_loader, desc=f"Epoch {i} - Validation"):
            data_v, label_v = data_v.to(device), label_v.to(device)
            logits_v = model(data_v)
            loss_v = loss_fn(logits_v, label_v)
            val_loss += loss_v.item()
            preds = torch.softmax(logits_v, dim=-1).argmax(dim=-1)
            all_val_pred.append(preds.cpu())
            all_val_label.append(label_v.cpu())
            val_acc += (preds == label_v).float().mean().item()

    val_acc /= len(val_loader)
    val_loss /= len(val_loader)

    if is_master:
        print(f"[Validation] Epoch {i}: Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    if val_acc > max_acc:
        max_acc = val_acc
        trigger_times = 0
        if is_master:
            save_ckpt(i, model, optimizer, scheduler, val_loss, model_name, ckpt_dir)
    else:
        trigger_times += 1
        if trigger_times >= PATIENCE:
            if is_master:
                print("[EarlyStopping] Triggered. Stopping training.")
            break

model.eval()
all_embeddings = []
all_labels = []

with torch.no_grad():
    for data_v, labels_v in tqdm(val_loader, desc="Extracting validation embeddings after training"):
        data_v, labels_v = data_v.to(device), labels_v.to(device)
        _ = model(data_v)
        conv_feats = model.module.to_out.current_conv_output
        all_embeddings.append(conv_feats)
        all_labels.append(labels_v.cpu())

all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
all_labels = torch.cat(all_labels, dim=0).numpy()

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=SEED)
embedding_umap = reducer.fit_transform(all_embeddings)

plt.figure(figsize=(8,6))
scatter = plt.scatter(embedding_umap[:, 0], embedding_umap[:, 1], c=all_labels, cmap='Spectral', s=5)
plt.colorbar(scatter)
plt.title("UMAP projection of validation embeddings (after all epochs)")
plt.show()

import os
import argparse
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from performer_pytorch import PerformerLM
from utils import *
import scanpy as sc
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import random

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", "--local-rank", type=int, default=-1, help='Local process rank.')
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
parser.add_argument("--epoch", type=int, default=10, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--batch_size", type=int, default=3, help='Number of batch size.')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
parser.add_argument("--data_path", type=str, default='./data/Zheng68K.h5ad', help='Path of data for finetune.')
parser.add_argument("--model_path", type=str, default='./panglao_pretrained.pth', help='Path of pretrained model.')
parser.add_argument("--ckpt_dir", type=str, default='./ckpts/', help='Directory of checkpoint to save.')
parser.add_argument("--model_name", type=str, default='finetune', help='Finetuned model name.')
args = parser.parse_args()

# Set seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num + 1
VALIDATE_EVERY = args.valid_every
UNASSIGN_THRES = 0.0
PATIENCE = 10

model_name = args.model_name
ckpt_dir = args.ckpt_dir

local_rank = args.local_rank
is_master = local_rank == 0

# DDP setup
dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
world_size = torch.distributed.get_world_size()

if is_master:
    print(f"[Init] Seed: {SEED}, Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"[Init] Using {world_size} GPUs, local_rank: {local_rank}")

CLASS = args.bin_num + 2
SEQ_LEN = args.gene_num + 1

class SCDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        full_seq = self.data[index].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0])))
        if index < 3 and is_master:
            print(f"[Dataset] idx {index}, sequence sample (first 10): {full_seq[:10]}, label: {self.label[index]}")
        return full_seq, self.label[index]

    def __len__(self):
        return self.data.shape[0]

class Identity(nn.Module):
    def __init__(self, dropout=0., h_dim=100, out_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(SEQ_LEN, 512)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, h_dim)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(h_dim, out_dim)
        self._printed = False

    def forward(self, x):
        if not self._printed:
            print(f"Shape after Performer, before Identity: {x.shape}")
        x = x[:, None, :, :]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        if not self._printed:
            print(f"Shape after Identity: {x.shape}")
            self._printed = True
        return x

data = sc.read_h5ad(args.data_path)
label_dict, label = np.unique(data.obs['celltype'], return_inverse=True)
label = torch.from_numpy(label)
data = data.X

if is_master:
    os.makedirs(args.ckpt_dir, exist_ok=True)
    with open(os.path.join(args.ckpt_dir, 'label_dict.pkl'), 'wb') as f:
        pkl.dump(label_dict, f)
    with open(os.path.join(args.ckpt_dir, 'label.pkl'), 'wb') as f:
        pkl.dump(label, f)
    print(f"[Data] Loaded data shape: {data.shape}, Labels: {len(label_dict)}")

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
for train_idx, val_idx in sss.split(data, label):
    train_dataset = SCDataset(data[train_idx], label[train_idx])
    val_dataset = SCDataset(data[val_idx], label[val_idx])

if is_master:
    print(f"[Split] Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

model = PerformerLM(
    num_tokens=CLASS,
    dim=200,
    depth=6,
    max_seq_len=SEQ_LEN,
    heads=10,
    g2v_position_emb=args.pos_embed
)

ckpt = torch.load(args.model_path, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
for param in model.parameters():
    param.requires_grad = False

model.to_out = Identity(dropout=0., h_dim=128, out_dim=len(label_dict))
model.add_module("to_out", model.to_out)
for param in model.to_out.parameters():
    param.requires_grad = True

model = model.to(device)
model = DDP(model, device_ids=[local_rank])

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
if is_master:
    print(f"[Model] Loaded pretrained weights. Trainable params: {trainable_params}")

criterion = nn.CrossEntropyLoss().to(local_rank)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                         sampler=train_sampler, num_workers=0, pin_memory=True)

val_sampler = DistributedSampler(val_dataset)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True) if is_master else None

dist.barrier()
trigger_times = 0
max_acc = 0.0

if is_master:
    print("[Training] Starting training...")

for i in range(1, EPOCHS+1):
    train_loader.sampler.set_epoch(i)
    model.train()
    dist.barrier()

    running_loss = 0.0
    cum_acc = 0.0
    for index, (data, labels) in enumerate(train_loader):
        index += 1
        data, labels = data.to(device), labels.to(device)

        if index % GRADIENT_ACCUMULATION != 0:
            with model.no_sync():
                logits = model(data)
                loss = criterion(logits, labels)
                loss.backward()
        else:
            logits = model(data)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
            optimizer.step()
            optimizer.zero_grad()
            if is_master:
                print(f"[Epoch {i}] Batch {index} step done, loss: {loss.item():.6f}")

        running_loss += loss.item()
        softmax = nn.Softmax(dim=-1)
        final = softmax(logits).argmax(dim=-1)
        pred_num = labels.size(0)
        correct_num = torch.eq(final, labels).sum().item()
        cum_acc += correct_num / pred_num

    epoch_loss = running_loss / index
    epoch_acc = 100 * cum_acc / index
    epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
    epoch_acc = get_reduced(epoch_acc, local_rank, 0, world_size)

    if is_master:
        print(f'[Epoch {i}] Train Loss: {epoch_loss:.6f} | Train Accuracy: {epoch_acc:.2f}%')
    dist.barrier()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        dist.barrier()
        running_loss = 0.0
        predictions, truths = [], []
        with torch.no_grad():
            for index, (data_v, labels_v) in enumerate(val_loader):
                index += 1
                data_v, labels_v = data_v.to(device), labels_v.to(device)
                logits = model(data_v)
                loss = criterion(logits, labels_v)
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
            print(f'[Epoch {i}] Val Loss: {val_loss:.6f} | Val F1: {f1:.4f}')
            print(confusion_matrix(truths, predictions))
            print(classification_report(truths, predictions, target_names=label_dict.tolist(), digits=4))

        if cur_acc > max_acc:
            max_acc = cur_acc
            trigger_times = 0
            save_best_ckpt(i, model, optimizer, val_loss, model_name, ckpt_dir)
            if is_master:
                print(f'[Epoch {i}] New best model saved (Acc: {cur_acc:.4f})')
        else:
            trigger_times += 1
            if trigger_times > PATIENCE:
                if is_master:
                    print(f"[Early Stop] Triggered at Epoch {i}")
                break

        del predictions, truths

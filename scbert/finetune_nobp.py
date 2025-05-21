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
import scanpy as sc
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import random

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
parser.add_argument("--bin_num", type=int, default=5)
parser.add_argument("--gene_num", type=int, default=16906)
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--batch_size", type=int, default=3)
parser.add_argument("--pos_embed", type=bool, default=True)
parser.add_argument("--data_path", type=str, default='./data/Zheng68K.h5ad')
parser.add_argument("--model_path", type=str, default='./panglao_pretrained.pth')
parser.add_argument("--ckpt_dir", type=str, default='./ckpts/')
parser.add_argument("--model_name", type=str, default='finetune_ready')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

local_rank = args.local_rank
is_master = local_rank == 0
dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

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
        #print(f"[Dataset] Index {index}, Label: {label}")
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

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
for train_idx, val_idx in sss.split(data, label):
    train_dataset = SCDataset(data[train_idx], label[train_idx])
    val_dataset = SCDataset(data[val_idx], label[val_idx])

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
model.to_out.requires_grad_(True)
model = model.to(device)
model = DDP(model, device_ids=[local_rank])

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                         sampler=train_sampler,num_workers=0, pin_memory=True)

val_sampler = DistributedSampler(val_dataset)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                       sampler=val_sampler,num_workers=0, pin_memory=True)

for i, (x, y) in enumerate(train_loader):
    #print(f"Batch {i}, x shape: {x.shape}, y shape: {y.shape}")
    if i == 1:
        break

for epoch in range(10):
    #print(f"[Rank {local_rank}] Starting epoch {epoch}")

    model.train()
    train_sampler.set_epoch(epoch)
    #print(f"[Rank {local_rank}] Beginning epoch {epoch} with {len(train_loader)} batches")

    total_loss = 0.0
    
    for batch_idx, (data_train, labels_train) in enumerate(train_loader):
        #print(f"[Rank {local_rank}] Epoch {epoch} - Batch {batch_idx}")
        try:
            data_train = data_train.to(device)
            labels_train = labels_train.to(device)

            optimizer.zero_grad()
            logits = model(data_train)
            #print(f"logits shape: {logits.shape}, labels shape: {labels_train.shape}")
            loss = criterion(logits, labels_train)
            #print(f"loss: {loss.item()}")
            loss.backward()
            #print("backward done")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        except Exception as e:
            #print(f"[Rank {local_rank}] Exception at batch {batch_idx}: {e}")
            raise

    
    if is_master:
        print(f"Epoch {epoch+1}/10, Loss: {total_loss / len(train_loader):.4f}")

if is_master:
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data_val, labels_val in val_loader:
            data_val = data_val.to(device)
            labels_val = labels_val.to(device)
            logits = model(data_val)
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(labels_val.cpu())

    #print(f"Logits shape: {logits.shape}")

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\nValidation Results on Full Dataset:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))

    save_path = os.path.join(args.ckpt_dir, f"{args.model_name}.pth")
    torch.save({
        'model_state_dict': model.module.state_dict(),
        'label_dict': label_dict,
        'args': vars(args)
    }, save_path)
    print(f"\nModel successfully saved to {save_path}")

dist.barrier()
dist.destroy_process_group()
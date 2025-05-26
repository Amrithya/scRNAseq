import os
import argparse
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.benchmark as benchmark
import torch.distributed as dist
from performer_pytorch import PerformerLM
from utils import *
import scanpy as sc
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
import random
import socket
import traceback
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--bin_num", type=int, default=5)
parser.add_argument("--gene_num", type=int, default=16906)
parser.add_argument("--epoch", type=int, default=3)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--grad_acc", type=int, default=60)
parser.add_argument("--valid_every", type=int, default=1)
parser.add_argument("--pos_embed", type=bool, default=True)
parser.add_argument("--data_path", type=str, default='./data/Zheng68K.h5ad')
parser.add_argument("--model_path", type=str, default='./panglao_pretrained.pth')
parser.add_argument("--ckpt_dir", type=str, default='./ckpts/')
parser.add_argument("--model_name", type=str, default='finetune')
args, _ = parser.parse_known_args()

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

local_rank = int(os.environ.get("LOCAL_RANK", -1))
if local_rank == -1:
    raise ValueError("LOCAL_RANK env var missing")

dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device(local_rank)
world_size = dist.get_world_size()
is_master = local_rank == 0

rank = int(os.environ.get("RANK", -1))
local_rank = int(os.environ.get("LOCAL_RANK", -1))
world_size = int(os.environ.get("WORLD_SIZE", -1))

try:
    torch.cuda.set_device(local_rank)
except Exception as e:
    print(f"[Rank {rank}] Failed to set device: {e}")

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
        return full_seq, self.label[index]
    def __len__(self):
        return self.data.shape[0]

class Identity(nn.Module):
    def __init__(self, dropout=0., h_dim=100, out_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(SEQ_LEN, out_dim)
        self.act1 = nn.ReLU()
    def forward(self, x):
        x = x[:, None, :, :]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.act1(x)
        return x

def save_checkpoint(epoch, model, optimizer, loss, name, path):
    if isinstance(model, DDP):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, os.path.join(path, f"{name}_checkpoint.pth"))

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
model.add_module("to_out", model.to_out)
for param in model.to_out.parameters():
    param.requires_grad = True

model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

start_epoch = 1
ckpt_path = os.path.join(args.ckpt_dir, f"{args.model_name}_checkpoint.pth")
if os.path.exists(ckpt_path):
    if is_master:
        print(f"[Resuming] Loading checkpoint from {ckpt_path}")
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    if is_master:
        print(f"[Resuming] Resuming from epoch {start_epoch}")

train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=0, pin_memory=True)
val_sampler = DistributedSampler(val_dataset)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=0, pin_memory=True)

dist.barrier()

max_acc = 0.0
trigger_times = 0

try:
    for epoch in range(start_epoch, args.epoch + 1):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        dist.barrier()

        if is_master:
            print(f"\n[Epoch {epoch}] Training started")

        running_loss = 0.0
        cum_acc = 0.0

        data_iter = tqdm(train_loader, desc=f"[Epoch {epoch}] Training", disable=not is_master)

        for step, (data, labels) in enumerate(data_iter, 1):
            data, labels = data.to(device), labels.to(device)

            try:
                if step % args.grad_acc != 0:
                    with model.no_sync():
                        logits = model(data)
                        loss = criterion(logits, labels)
                        loss.backward()
                else:
                    logits = model(data)
                    loss = criterion(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
                    optimizer.step()
                    optimizer.zero_grad()
            except Exception as e:
                print(f"[Rank {rank}] Error during backprop at step {step}: {e}")
                traceback.print_exc()
                continue

            running_loss += loss.item()
            preds = torch.softmax(logits, dim=-1).argmax(dim=-1)
            cum_acc += (preds == labels).sum().item() / labels.size(0)

        avg_loss = get_reduced(running_loss / step, local_rank, 0, world_size)
        avg_acc = get_reduced(100 * cum_acc / step, local_rank, 0, world_size)

        if is_master:
            print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.2f}%")

        if epoch % args.valid_every == 0:
            model.eval()
            dist.barrier()

            running_loss = 0.0
            predictions, truths = [], []

            val_iter = tqdm(val_loader, desc=f"[Epoch {epoch}] Validation", disable=not is_master)

            with torch.no_grad():
                for step, (data_v, labels_v) in enumerate(val_iter, 1):
                    data_v, labels_v = data_v.to(device), labels_v.to(device)

                    try:
                        logits = model(data_v)
                        loss = criterion(logits, labels_v)
                    except Exception as e:
                        print(f"[Rank {rank}] Error during validation at step {step}: {e}")
                        traceback.print_exc()
                        continue

                    running_loss += loss.item()

                    prob = torch.softmax(logits, dim=-1)
                    final = prob.argmax(dim=-1)
                    max_probs, _ = prob.max(dim=-1)
                    final[max_probs < UNASSIGN_THRES] = -1

                    predictions.append(final.cpu())
                    truths.append(labels_v.cpu())

            predictions = distributed_concat(torch.cat(predictions), len(val_dataset), world_size)
            truths = distributed_concat(torch.cat(truths), len(val_dataset), world_size)

            if is_master:
                mask = predictions != -1
                filtered_preds = predictions[mask].cpu().numpy()
                filtered_truths = truths[mask].cpu().numpy()

                acc = accuracy_score(filtered_truths, filtered_preds)
                f1 = f1_score(filtered_truths, filtered_preds, average='macro')
                val_loss = get_reduced(running_loss / step, local_rank, 0, world_size)

                print(f"[Epoch {epoch}] Validation Loss: {val_loss:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

                if acc > max_acc:
                    max_acc = acc
                    trigger_times = 0
                    save_checkpoint(epoch, model, optimizer, val_loss, args.model_name, args.ckpt_dir)
                    print(f"[Epoch {epoch}] New best model saved with acc = {acc:.4f}")
                else:
                    trigger_times += 1
                    print(f"[Epoch {epoch}] No improvement. Trigger times: {trigger_times}/{PATIENCE}")
                    if trigger_times > PATIENCE:
                        print("[Early Stop] Trigger patience reached.")
                        break

            dist.barrier()

except KeyboardInterrupt:
    if is_master:
        print("[Interrupted] Training manually stopped.")
except Exception as e:
    print(f"[Rank {rank}] Unhandled exception during training: {e}")
    traceback.print_exc()
finally:
    dist.destroy_process_group()
    if is_master:
        print("DDP training process finished.")

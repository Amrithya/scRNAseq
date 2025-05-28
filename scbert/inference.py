import os
import argparse
import numpy as np
import torch
from performer_pytorch import PerformerLM
import scanpy as sc

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
parser.add_argument("--bin_num", type=int, default=5)
parser.add_argument("--gene_num", type=int, default=16906)
parser.add_argument("--epoch", type=int, default=23)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--grad_acc", type=int, default=60)
parser.add_argument("--valid_every", type=int, default=1)
parser.add_argument("--pos_embed", type=bool, default=True)
parser.add_argument("--data_path", type=str, default='/data1/data/corpus/Zheng68K.h5ad')
parser.add_argument("--model_path", type=str, default='/data1/data/corpus/panglao_pretrain.pth')
parser.add_argument("--ckpt_dir", type=str, default='./ckpts/')
parser.add_argument("--model_name", type=str, default='finetune')
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

# Setup
SEQ_LEN = args.gene_num + 1
CLASS = args.bin_num + 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
adata = sc.read_h5ad(args.data_path)
X = adata.X
if hasattr(X, 'toarray'):
    X = X.toarray()

X_tokens = (X / X.max() * (CLASS - 1)).astype(np.int64)

# Model
model = PerformerLM(
    num_tokens=CLASS,
    dim=200,
    depth=6,
    max_seq_len=SEQ_LEN,
    heads=10,
    local_attn_heads=0,
    g2v_position_emb=args.pos_embed
)

ckpt = torch.load(args.model_path, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()

# Inference
all_hidden = []
with torch.no_grad():
    for start_idx in range(0, X_tokens.shape[0], args.batch_size):
        end_idx = min(start_idx + args.batch_size, X_tokens.shape[0])
        batch_tokens_np = X_tokens[start_idx:end_idx, :SEQ_LEN]
        batch_tokens = torch.tensor(batch_tokens_np, dtype=torch.long).to(device)

        embedded = model.token_emb(batch_tokens).float()
        if hasattr(model, 'pos_emb') and model.pos_emb is not None:
            embedded += model.pos_emb(embedded)

        hidden = model.performer(embedded)
        all_hidden.append(hidden.cpu())

# Save final output
all_hidden_tensor = torch.cat(all_hidden, dim=0)
print("Final shape of all hidden representations:", all_hidden_tensor.shape)
torch.save(all_hidden_tensor, 'performer_all_learned_representations.pt')

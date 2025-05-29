import os
import argparse
import numpy as np
import torch
from performer_pytorch import PerformerLM
import scanpy as sc
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--bin_num", type=int, default=5)
parser.add_argument("--gene_num", type=int, default=16906)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--pos_embed", type=bool, default=True)
parser.add_argument("--data_path", type=str, default='/data1/data/corpus/Zheng68K.h5ad')
parser.add_argument("--model_path", type=str, default='/data1/data/corpus/panglao_pretrain.pth')
parser.add_argument("--output_path", type=str, default='performer_cls_representations.pt')
args = parser.parse_args()

SEQ_LEN = args.gene_num + 1
CLASS = args.bin_num + 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adata = sc.read_h5ad(args.data_path)
X = adata.X
if hasattr(X, 'toarray'):
    X = X.toarray()

X_tokens = (X / X.max() * (CLASS - 1)).astype(np.int64)

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

all_cls_reps = []
with torch.no_grad():
    for start_idx in tqdm(range(0, X_tokens.shape[0], args.batch_size), desc="Running inference"):
        end_idx = min(start_idx + args.batch_size, X_tokens.shape[0])
        batch_tokens_np = X_tokens[start_idx:end_idx, :SEQ_LEN]
        batch_tokens = torch.tensor(batch_tokens_np, dtype=torch.long).to(device)
        embedded = model.token_emb(batch_tokens).float()
        if hasattr(model, 'pos_emb') and model.pos_emb is not None:
            embedded += model.pos_emb(embedded)
        hidden = model.performer(embedded)
        cls_hidden = hidden[:, 0, :]
        all_cls_reps.append(cls_hidden.cpu())
        torch.cuda.empty_cache()

all_hidden_tensor = torch.cat(all_cls_reps, dim=0)
print("Final shape of CLS hidden representations:", all_hidden_tensor.shape)
torch.save(all_hidden_tensor, args.output_path)

import os
import argparse
import numpy as np
import torch
from performer_pytorch import PerformerLM
import scanpy as sc

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


SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num + 1
VALIDATE_EVERY = args.valid_every

CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed
model_name = args.model_name
ckpt_dir = args.ckpt_dir

adata = sc.read_h5ad(args.data_path)
X = adata.X
if hasattr(X, 'toarray'):
    X = X.toarray()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_tokens = (X / X.max() * (CLASS - 1)).astype(np.int64)
input_tokens_np = X_tokens[:BATCH_SIZE, :SEQ_LEN]
input_tokens = torch.tensor(input_tokens_np, dtype=torch.long).to(device)

model = PerformerLM(
    num_tokens=CLASS,
    dim=200,
    depth=6,
    max_seq_len=SEQ_LEN,
    heads=10,
    local_attn_heads=0,
    g2v_position_emb=POS_EMBED_USING
)
ckpt = torch.load(args.model_path, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()

with torch.no_grad():
    hidden = model(input_tokens)

print("Hidden representation shape:", hidden.shape)
learned_representations = hidden.cpu()
torch.save(learned_representations, 'performer_learned_representations.pt')

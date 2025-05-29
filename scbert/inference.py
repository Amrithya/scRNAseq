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
parser.add_argument("--output_path", type=str, default='/data1/data/corpus/performer_representations')
args = parser.parse_args()

SEQ_LEN = args.gene_num + 1
CLASS = args.bin_num + 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adata = sc.read_h5ad(args.data_path)
data_tensor = torch.tensor(adata.X.toarray()).float()

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

all_outputs = []
chunk_size = 1000
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

with torch.no_grad():
    for i in tqdm(range(0, len(data_tensor), args.batch_size)):
        batch = data_tensor[i:i + args.batch_size].to(device)
        output = model(batch)
        cls_output = output[:, 0, :]
        all_outputs.append(cls_output.cpu())
        if len(all_outputs) * args.batch_size >= chunk_size:
            chunk = torch.cat(all_outputs, dim=0)
            torch.save(chunk, f'{args.output_path}_chunk_{i // chunk_size}.pt')
            all_outputs = []

if len(all_outputs) > 0:
    final_chunk = torch.cat(all_outputs, dim=0)
    torch.save(final_chunk, f'{args.output_path}_final.pt')

print("All outputs saved successfully!")

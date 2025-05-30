import os
import argparse
import numpy as np
import torch
import torch.nn as nn
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
parser.add_argument("--output_path", type=str, default='/data1/data/corpus/conv1d_representations.npy')
args = parser.parse_args()

SEQ_LEN = args.gene_num + 1
CLASS = args.bin_num + 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adata = sc.read_h5ad(args.data_path)
expression = adata.X.toarray()
bin_edges = np.histogram_bin_edges(expression, bins=args.bin_num)
tokenized = np.digitize(expression, bins=bin_edges, right=False)
cls_token = np.zeros((tokenized.shape[0], 1), dtype=np.int64)
tokenized = np.hstack([cls_token, tokenized])
data_tensor = torch.tensor(tokenized, dtype=torch.long)

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

conv1 = nn.Conv2d(1, 1, (1, 200)).to(device)

all_outputs = []
with torch.no_grad():
    for i in tqdm(range(0, len(data_tensor), args.batch_size)):
        batch = data_tensor[i:i + args.batch_size].to(device)
        output = model(batch)  # (B, S, D) 
        output = output.unsqueeze(1)  
        reduced = conv1(output).squeeze(1).squeeze(-1)  # (B, S)
        all_outputs.append(reduced.cpu())

final_output = torch.cat(all_outputs, dim=0).numpy()  # (N, S)
print("Final output shape:", final_output.shape)

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
np.save(args.output_path, final_output)

print("All outputs saved successfully!")

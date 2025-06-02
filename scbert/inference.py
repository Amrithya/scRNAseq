import os
import argparse
import numpy as np
import torch
from performer_pytorch import PerformerLM
import scanpy as sc
from tqdm import tqdm
import umap.umap_ as umap
import matplotlib.pyplot as plt

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

def tokn_expression(expr, num_classes):
    expr_min = expr.min(axis=1, keepdims=True)
    expr_max = expr.max(axis=1, keepdims=True)
    expr_norm = (expr - expr_min) / (expr_max - expr_min + 1e-8)
    tokens = (expr_norm * (num_classes - 1)).astype(np.int64)
    return tokens

X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
tokens = tokn_expression(X, CLASS)

cls_token = np.zeros((tokens.shape[0], 1), dtype=np.int64)
tokens = np.concatenate([cls_token, tokens], axis=1)

batch_size = args.batch_size
all_cls_embeddings = []

for i in tqdm(range(0, tokens.shape[0], batch_size)):
    batch_tokens = tokens[i:i+batch_size]
    batch_tokens = torch.tensor(batch_tokens).to(device)

    with torch.no_grad():
        output = model(batch_tokens)
        cls_embedding = output[:, 0, :].cpu().numpy()
        all_cls_embeddings.append(cls_embedding)

all_cls_embeddings = np.vstack(all_cls_embeddings)
np.save(args.output_path, all_cls_embeddings)

reducer = umap.UMAP()
embedding_2d = reducer.fit_transform(all_cls_embeddings)

labels = adata.obs.get('label')
if labels is None:
    labels = np.zeros(len(embedding_2d))

plt.figure(figsize=(8, 6))
scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels.astype('category').cat.codes if hasattr(labels, 'cat') else labels, cmap='tab10', s=5)
plt.colorbar(scatter)
plt.title('UMAP of CLS token embeddings')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.tight_layout()
plt.show()

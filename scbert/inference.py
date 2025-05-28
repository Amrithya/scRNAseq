import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from performer_pytorch import PerformerLM
import scanpy as sc

class SCDataset(Dataset):
    def __init__(self, data, max_val):
        self.data = data
        self.max_val = max_val

    def __getitem__(self, index):
        full_seq = self.data[index].toarray()[0]
        full_seq[full_seq > self.max_val] = self.max_val
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0])))
        return full_seq

    def __len__(self):
        return self.data.shape[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_num", type=int, default=5)
    parser.add_argument("--gene_num", type=int, default=16906)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pos_embed", type=bool, default=True)
    parser.add_argument("--data_path", type=str, default="/data1/data/corpus/Zheng68K.h5ad")
    parser.add_argument("--model_path", type=str, default="/data1/data/corpus/panglao_pretrain.pth")
    parser.add_argument("--output_path", type=str, default="./inference_embeddings.npy")
    args = parser.parse_args()

    CLASS = args.bin_num + 2
    SEQ_LEN = args.gene_num + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adata = sc.read_h5ad(args.data_path)
    dataset = SCDataset(adata.X, max_val=CLASS - 2)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = PerformerLM(
        num_tokens=CLASS,
        dim=200,
        depth=6,
        max_seq_len=SEQ_LEN,
        heads=10,
        g2v_position_emb=args.pos_embed
    )
    ckpt = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    all_embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            token_embeddings = model.token_emb(batch)
            hidden_states = model.transformer(token_embeddings)
            hidden_states = hidden_states[:, :-1, :]
            embeddings_2d = hidden_states.mean(dim=2)
            all_embeddings.append(embeddings_2d.cpu())

    all_embeddings = torch.cat(all_embeddings).numpy()
    np.save(args.output_path, all_embeddings)
    print(f"Saved embeddings to {args.output_path}, shape: {all_embeddings.shape}")

if __name__ == "__main__":
    main()

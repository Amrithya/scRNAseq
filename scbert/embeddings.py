import torch
from performer_pytorch import PerformerLM
import numpy as np

CLASS = 7  
SEQ_LEN = 16907

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PerformerLM(
    num_tokens=CLASS,
    dim=200,
    depth=6,
    max_seq_len=SEQ_LEN,
    heads=10,
    g2v_position_emb=True
)
model.load_state_dict(torch.load("/data1/data/corpus/panglao_pretrain.pth", map_location="cpu")['model_state_dict'])
model.to(device)
model.eval()

dummy_input = torch.zeros((1, SEQ_LEN), dtype=torch.long).to(device)

with torch.no_grad():
    out = model(dummy_input)
    embeddings = out[:, -1, :].cpu().numpy()

np.save("perfomer_embeddings.npy", embeddings)
print("Saved embeddings shape:", embeddings.shape)

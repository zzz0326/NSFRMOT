import os
import glob
from collections import defaultdict
import cv2
import random
import torch
from PIL import Image
from torchvision import transforms
import open_clip
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import math
from collections import Counter

class ClipTrackDataset(Dataset):
    def __init__(self, embedding_path: str, global_id_path: str, class_id_path: str):
        self.embeddings = np.load(embedding_path)        # shape: (N, D)
        self.global_ids = np.load(global_id_path)        # shape: (N,)
        self.class_ids = np.load(class_id_path)          # shape: (N,)

        assert len(self.embeddings) == len(self.global_ids) == len(self.class_ids)


        self.track_dict = defaultdict(list)
        for i, gid in enumerate(self.global_ids):
            self.track_dict[int(gid)].append(i)


        self.track_ids = list(self.track_dict.keys())

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        indices = self.track_dict[track_id]

        embs = self.embeddings[indices]              # shape: (T, D)
        class_ids = self.class_ids[indices]          # shape: (T,)
        global_ids = self.global_ids[indices]        # shape: (T,)

        return embs, global_ids, class_ids

dataset = ClipTrackDataset("", "", "/")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

dataset_val = ClipTrackDataset("", "", "")
dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=True)



prompts = [f"a photo of {obj}" for obj in car_list]
model = open_clip.create_model('ViT-L-14', pretrained=None, device='cuda')
state_dict = torch.load("",
                        map_location='cuda')

model.load_state_dict(state_dict)
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-L-14')
tokenized = tokenizer(prompts).to('cuda')

with torch.no_grad():
    text_embs = model.encode_text(tokenized)
text_embs = text_embs.unsqueeze(1)  # (32, 1, 768)
text_embs = F.interpolate(text_embs, size=128, mode='linear', align_corners=True)  # (32, 1, 128)
text_embs = text_embs.squeeze(1)

class CrossAttentionClassifier(nn.Module):
    def __init__(self, dim, num_classes, heads=4):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.classifier = nn.Linear(dim, num_classes)  # 输出类别概率

    def forward(self, query_emb, text_embs):

        Q = self.q_proj(query_emb.unsqueeze(0))       # (1, 1, d)
        K = self.k_proj(text_embs.unsqueeze(0))       # (1, C, d)
        V = self.v_proj(text_embs.unsqueeze(0))       # (1, C, d)

        out, _ = self.attn(Q, K, V)                   # (1, 1, d)
        out = out.squeeze(0).squeeze(0)               # (d,)
        logits = self.classifier(out)                 # (C,)
        return logits


def compute_loss(track_embs, text_embs, model, k=5, loss_type="bce", n_hard=5, val=False):

    C = text_embs.size(0)


    avg_emb = track_embs.mean(dim=0, keepdim=True) # (1, d)

    sim = torch.matmul(F.normalize(text_embs, dim=-1), F.normalize(avg_emb, dim=-1).T).squeeze(1)
    topk_indices = torch.topk(sim, k=k).indices  # (k,) → label

    sim_to_avg = F.cosine_similarity(track_embs, avg_emb)  # (w,)
    hard_indices = torch.argsort(sim_to_avg)[:n_hard]
    hard_embs = track_embs[hard_indices]  # (n, d)


    total_loss = 0.
    for query in hard_embs:
        if val == True:
            with torch.no_grad():
                logits = model(query.unsqueeze(0), text_embs)
        else:
            logits = model(query.unsqueeze(0), text_embs)


        if loss_type == "bce":
            label = torch.zeros(C, device=logits.device)
            label[topk_indices[0]] = 1.0
            label[topk_indices[1]] = 0.8
            label[topk_indices[2]] = 0.6
            label[topk_indices[3]] = 0.4
            label[topk_indices[4]] = 0.2
            loss = F.binary_cross_entropy_with_logits(logits, label)
        elif loss_type == "ce":

            label = topk_indices[0].unsqueeze(0)  # (1,)
            loss = F.cross_entropy(logits.unsqueeze(0), label)
        else:
            raise ValueError("Unsupported loss type: choose 'bce' or 'ce'")
        total_loss += loss

    return total_loss


def compute_loss_mu(track_embs, text_embs, model, k=5, loss_type="bce", n_hard=5, val=False):

    C = text_embs.size(0)


    avg_emb = track_embs.mean(dim=0, keepdim=True) # (1, d)

    sim = torch.matmul(F.normalize(text_embs, dim=-1), F.normalize(avg_emb, dim=-1).T).squeeze(1)
    topk_indices = torch.topk(sim, k=k).indices  # (k,) → label


    if val:
        with torch.no_grad():
            logits = model(track_embs, text_embs)  # (B, C)
    else:
        logits = model(track_embs, text_embs)  # (B, C)


    label = torch.zeros(C, device=logits.device)
    label[topk_indices[0]] = 1.0
    label[topk_indices[1]] = 0.8
    label[topk_indices[2]] = 0.6
    label[topk_indices[3]] = 0.4
    label[topk_indices[4]] = 0.2

    label = label.unsqueeze(0).expand(logits.size(0), -1)  # (B, C)



    prob = torch.sigmoid(logits)  # (B, C)


    pos_prob = (prob * label).sum(dim=1) / (label.sum() + 1e-6)  # (B,)

    mu = pos_prob.mean()
    var = ((pos_prob - mu) ** 2).mean()

    loss = -mu + var
        # total_loss += loss

    return loss

dim = 128
num_classes = 20
model = CrossAttentionClassifier(dim=dim, num_classes=num_classes).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


step = 0
select_cof = 0.2
for iiii in range(100):
    for embeddings, global_ids, class_ids in dataloader:
        embeddings = embeddings.cuda() # (N, D)
        embeddings = embeddings.squeeze(0).squeeze(1)
        # print(embeddings.shape)
        if embeddings.shape[0] > 30:
            step += 1
            loss = compute_loss_mu(embeddings, text_embs, model, k=5, n_hard=int(embeddings.shape[0]*select_cof))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                # pred_top1 = logits.argmax().item()
                print(f"Step {step} | Loss: {loss.item():.4f}")
            if step % 1000 == 0:
                loss_val = 0.
                for embeddings_val, global_ids, class_ids in dataloader_val:
                    embeddings_val = embeddings_val.cuda()  # (N, D)
                    embeddings_val = embeddings_val.squeeze(0).squeeze(1)
                    if embeddings_val.shape[0] > 30:
                        loss_val += compute_loss_mu(embeddings_val, text_embs, model, k=5, n_hard=int(embeddings.shape[0]*select_cof), val=True)
                ckpt_path = os.path.join('', f"u2mot_cross_replace_{int(loss_val)}_step{step}_mu.pt")
                torch.save(model.state_dict(), ckpt_path)
                # print(f"Saved model to {ckpt_path}")
                # pred_top1 = logits.argmax().item()
                print(f"Loss_val: {loss_val.item():.4f}")

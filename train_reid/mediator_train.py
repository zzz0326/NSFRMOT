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
from torchvision.ops import roi_align
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

    # 1. 计算平均 embedding
    avg_emb = track_embs.mean(dim=0, keepdim=True) # (1, d)
    # print(avg_emb.shape)
    # 2. 找到最相似的 k 个类别
    sim = torch.matmul(F.normalize(text_embs, dim=-1), F.normalize(avg_emb, dim=-1).T).squeeze(1)
    topk_indices = torch.topk(sim, k=k).indices  # (k,) → label
    # print(topk_indices)

    if val:
        with torch.no_grad():
            logits = model(track_embs, text_embs)  # (B, C)
    else:
        logits = model(track_embs, text_embs)  # (B, C)

    # 构造 soft label（保持你原来的 top-k 权重）
    label = torch.zeros(C, device=logits.device)
    label[topk_indices[0]] = 1.0
    label[topk_indices[1]] = 0.8
    label[topk_indices[2]] = 0.6
    label[topk_indices[3]] = 0.4
    label[topk_indices[4]] = 0.2

    label = label.unsqueeze(0).expand(logits.size(0), -1)  # (B, C)

    # ===== 新的 variance-based BCE 替代 =====

    prob = torch.sigmoid(logits)  # (B, C)

    # 只关注正类部分
    pos_prob = (prob * label).sum(dim=1) / (label.sum() + 1e-6)  # (B,)

    mu = pos_prob.mean()
    var = ((pos_prob - mu) ** 2).mean()

    loss = -mu + var
        # total_loss += loss

    return loss




def train(clip_path, save_path):

    dataset = ClipTrackDataset(save_path+"/M_embeddings.npy", save_path+"/M_global_ids.npy", save_path+"/M_class_ids.npy")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


    car_list = ['black Maria', 'jinrikisha', 'thresher', 'oxcart', 'barrow', 'tandem', 'thrasher', 'grille', 'ricksha', 'monocycle', 'rig', 'hack', 'wrecker', 'bullet', 'radiator grille', 'R.V.', 'beach waggon', 'unicycle', 'pinwheel', 'velocipede']

    prompts = [f"a photo of {obj}" for obj in car_list]
    model = open_clip.create_model('ViT-L-14', pretrained=None, device='cuda')
    state_dict = torch.load(clip_path,
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

    dim = 128
    num_classes = 20
    model = CrossAttentionClassifier(dim=dim, num_classes=num_classes).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    step = 0
    select_cof = 0.2
    for iiii in range(100):
        for embeddings, global_ids, class_ids in dataloader:
            embeddings = embeddings.cuda() # (N, D)
            embeddings = embeddings.squeeze(0).squeeze(1)
            if embeddings.shape[0] > 30:
                step += 1
                loss = compute_loss(embeddings, text_embs, model, k=5, n_hard=int(embeddings.shape[0]*select_cof))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 100 == 0:
                    # pred_top1 = logits.argmax().item()
                    print(f"Step {step} | Loss: {loss.item():.4f}")
                if step % 1000 == 0:
                    ckpt_path = os.path.join(save_path, f"replace_{int(loss)}_step{step}.pt")
                    torch.save(model.state_dict(), ckpt_path)


valid_classes = {1, 4, 5, 6, 9}
def collect_tracks_from_txt(directory):
    results = []
    global_id_counter = 0

    for txt_path in sorted(glob.glob(os.path.join(directory, '*.txt'))):
        seq_name = os.path.basename(txt_path).split('.')[0]
        track_dict = {}

        with open(txt_path, 'r') as f:
            for line in f:
                items = line.strip().split(',')
                if len(items) < 8:
                    continue  # 跳过无效行
                frame_id = int(items[0])
                track_id = int(items[1])
                x, y, w, h = map(float, items[2:6])
                class_id = int(items[-3])

                if class_id not in valid_classes:
                    continue

                # 分配 global_id
                if track_id not in track_dict:
                    track_dict[track_id] = global_id_counter
                    global_id_counter += 1
                global_id = track_dict[track_id]

                results.append((global_id, seq_name, frame_id, track_id, x, y, w, h, class_id))

    return results

def extract_region_feats(reid_feat, outputs, test_size):
    """
    reid_feat: torch.Tensor of shape (1, h, w, c)
    outputs: numpy array of shape (N, 4+), with boxes in [x1, y1, x2, y2] format
    test_size: list or tuple, [H_img, W_img] – detector input size
    """

    b, h, w, c = reid_feat.shape
    assert b == 1, "Only batch size 1 is supported"

    scale_x = w / test_size[1]
    scale_y = h / test_size[0]

    boxes_xyxy = outputs[:, :4].copy()  # [x1, y1, x2, y2] in test_size space
    boxes_xyxy[:, [0, 2]] *= scale_x
    boxes_xyxy[:, [1, 3]] *= scale_y


    boxes = torch.cat([
        torch.zeros((boxes_xyxy.shape[0], 1)),  # batch_idx = 0
        torch.from_numpy(boxes_xyxy)
    ], dim=1)  # shape: (N, 5)


    reid_feat_nchw = reid_feat.permute(0, 3, 1, 2).contiguous()  # [1, c, h, w]


    region_feats = roi_align(
        input=reid_feat_nchw.float(),
        boxes=boxes.float(),
        output_size=(1, 1),         # 全局特征
        spatial_scale=1.0,          # 已缩放，保持1.0
        aligned=True
    )
    region_feats = region_feats.squeeze(-1).squeeze(-1)  # (N, c)

    return region_feats

def extract_embedding(img_root, label_path, npy_path, save_path):
    image_root = img_root
    results = collect_tracks_from_txt(label_path)
    frame_dict = defaultdict(list)
    for entry in results:
        global_id, seq_name, frame_id, track_id, x, y, w, h, class_id = entry
        frame_dict[(seq_name, frame_id)].append((x, y, w, h, global_id, class_id))

    all_embeddings = []
    all_global_ids = []
    all_class_ids = []

    now_feat_path = None
    feat_sum = None
    for (seq_name, frame_id), boxes in tqdm(frame_dict.items(), desc="Processing frames"):
        img_path = os.path.join(image_root, seq_name, f"{frame_id:07d}.jpg")
        # print(img_path)
        if not os.path.exists(img_path):
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            continue
        feat_path = os.path.join(npy_path, seq_name + ".npy")

        if now_feat_path != feat_path:
            feat_sum = np.load(feat_path, allow_pickle=True).item()
            now_feat_path = feat_path
        # feat_sum = np.load(feat_path, allow_pickle=True).item()
        feat_frame = feat_sum[f"{int(frame_id):06d}"]
        feat_frame = torch.from_numpy(feat_frame).unsqueeze(0)
        # crops = []
        ids = []
        cls_ids = []
        emb_list = []
        for x, y, w, h, gid, cls_id in boxes:
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            bbox = np.array([x1, y1, x2, y2])
            bbox = np.expand_dims(bbox, axis=0)
            emb_t = extract_region_feats(feat_frame, bbox, (image.size[1], image.size[0]))
            emb_list.append(emb_t.numpy())

            ids.append(gid)
            cls_ids.append(cls_id)
        emb = np.array(emb_list)


        all_embeddings.append(emb)
        all_global_ids.extend(ids)
        all_class_ids.extend(cls_ids)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_global_ids = np.array(all_global_ids)
    all_class_ids = np.array(all_class_ids)
    np.save(os.path.join(save_path, "M_embeddings.npy"), all_embeddings)
    np.save(os.path.join(save_path, "M_global_ids.npy"), all_global_ids)
    np.save(os.path.join(save_path, "M_class_ids.npy"), all_class_ids)
    print(f"Saved {len(all_embeddings)} embeddings to " + save_path)

img_root = 'a'
label_path ='b'
npy_path ='c'
save_path ='d'
extract_embedding(img_root=img_root, label_path=label_path, npy_path=npy_path, save_path=save_path)
clip_path = 'e'
train(save_path=save_path, clip_path=clip_path)

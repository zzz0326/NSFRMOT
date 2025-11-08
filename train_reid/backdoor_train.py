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

class ClipEmbeddingFusionDataset(Dataset):
    def __init__(self, embedding_path, label_path, alpha=0.5):
        self.embedding_path = embedding_path
        self.label_path = label_path
        self.alpha = alpha

        # Load all embeddings and labels
        self.embeddings = np.load(embedding_path)  # shape: (N, 768)
        self.labels = np.load(label_path)  # shape: (N,)

        # Group embeddings by (frame_id, track_id)
        self.index_pairs = self._build_pairs()

    def _build_pairs(self):
        # Try to pair each label 0 (foreground) embedding with corresponding label 1 (background)
        pairs = []
        for i in range(len(self.labels)):
            if self.labels[i] == 0:
                for j in range(len(self.labels)):
                    if self.labels[j] == 1:
                        pairs.append((i, j))  # (foreground_index, background_index)
                        break  # only one background per foreground
        return pairs

    def __len__(self):
        return len(self.index_pairs)

    def __getitem__(self, idx):
        i, j = self.index_pairs[idx]
        embed_0 = torch.tensor(self.embeddings[i], dtype=torch.float32)  # label 0
        embed_1 = torch.tensor(self.embeddings[j], dtype=torch.float32)  # label 1

        return embed_0, embed_1

class BackgroundClassifier(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            # nn.Dropout(p=0.5),  # 添加在第一个 ReLU 之后
            nn.Linear(128, 128),
            nn.ReLU(),
            # nn.Dropout(p=0.5),  # 添加在第二个 ReLU 之后
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(p=0.5),  # 添加在第三个 ReLU 之后
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def project_orthogonal(a, b):
    proj = torch.sum(a * b, dim=1, keepdim=True) / (torch.norm(b, dim=1, keepdim=True) ** 2 + 1e-8)
    return a - proj * b

def train_classifier(embedding_path, label_path, save_path='/data/MOT/VisDrone-MOT/CLIP-crop/backdoor/', epochs=5,
                     batch_size=128, lr=1e-6):
    dataset = ClipEmbeddingFusionDataset(embedding_path, label_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BackgroundClassifier().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for embed_0, embed_1 in dataloader:
            embed_0 = F.normalize(embed_0, dim=-1)
            embed_1 = F.normalize(embed_1, dim=-1)
            clean_embed = project_orthogonal(embed_0, embed_1).cuda() # label=0
            clean_label = torch.zeros((clean_embed.shape[0], 1)).cuda()
            embed_1 = embed_1.cuda()
            confounder_label = torch.ones((clean_embed.shape[0], 1)).cuda()
            logits_0 = model(clean_embed)
            loss_0 = criterion(logits_0, clean_label)
            logits_1 = model(embed_1)
            loss_1 = criterion(logits_1, confounder_label)
            loss = loss_0+loss_1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * clean_embed.size(0)

        avg_train_loss = train_loss / len(dataset)

        print(
            f"[Classifier] Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(save_path, f'backdoor_classifier{int(avg_train_loss *100)}_epoch_{epoch + 1}.pt'))

    return model

def compute_overlap_ratio(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_w, inter_h = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    area1 = w1 * h1
    return 0 if area1 == 0 else inter_area / area1

def extract_low_visibility_pairs(txt_path, threshold=0.3):
    track_data = defaultdict(dict)  # {track_id: {frame_id: bbox}}
    filename = txt_path.split('/')[-1].split('.')[0]

    with open(txt_path, 'r') as f:
        for line in f:
            items = line.strip().split(',')
            if len(items) < 6:
                continue
            frame_id = int(items[0])
            track_id = int(items[1])
            x, y, w, h = map(float, items[2:6])
            track_data[track_id][frame_id] = [x, y, w, h]

    results = []


    for track_id, frames in track_data.items():
        frame_ids = list(frames.keys())
        for fid1 in frame_ids:
            bbox1 = frames[fid1]
            for fid2 in frame_ids:
                if fid1 == fid2:
                    continue
                if fid2 not in frames:
                    results.append((fid1, track_id, fid2, 0.0, bbox1))
                else:
                    bbox2 = frames[fid2]
                    ratio = compute_overlap_ratio(bbox1, bbox2)
                    if ratio < threshold:
                        results.append((fid1, track_id, fid2, ratio, bbox1, filename))
    return results

def process_directory(directory, threshold=0.3):
    all_results = []
    txt_files = glob.glob(os.path.join(directory, '*.txt'))
    for txt_file in txt_files:
        results = extract_low_visibility_pairs(txt_file, threshold)
        all_results.extend(results)
    return all_results

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
        output_size=(1, 1),
        spatial_scale=1.0,
        aligned=True
    )
    region_feats = region_feats.squeeze(-1).squeeze(-1)  # (N, c)

    return region_feats

def extract_embedding(img_root, label_path, npy_path, save_path):
    image_root = img_root
    results = process_directory(label_path, threshold=1.0)
    best_pairs = {}
    for frame_id, track_id, other_frame_id, ratio, bbox, filename in results:
        key = (frame_id, track_id)
        if key not in best_pairs or ratio < best_pairs[key][0]:
            best_pairs[key] = (ratio, other_frame_id, bbox, filename)

    embeddings = []
    labels = []
    now_feat_path = None
    feat_sum = None
    for (frame_id, track_id), (ratio, other_frame_id, bbox, filename) in tqdm(best_pairs.items()):
        frame_path = os.path.join(image_root, filename, f"{int(frame_id):07d}.jpg")
        other_path = os.path.join(image_root, filename, f"{int(other_frame_id):07d}.jpg")

        feat_path = os.path.join(npy_path, filename+".npy")
        if now_feat_path != feat_path:
            feat_sum = np.load(feat_path, allow_pickle=True).item()
            now_feat_path = feat_path
        feat_frame = feat_sum[f"{int(frame_id):06d}"]
        feat_other = feat_sum[f"{int(other_frame_id):06d}"]
        feat_frame = torch.from_numpy(feat_frame).unsqueeze(0)
        feat_other = torch.from_numpy(feat_other).unsqueeze(0)
        if not os.path.exists(frame_path) or not os.path.exists(other_path):
            continue

        try:
            img1 = Image.open(frame_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image: {e}")
            continue
        # print(bbox)
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        bbox = np.array(bbox)
        bbox = np.expand_dims(bbox, axis=0)
        emb1 = extract_region_feats(feat_frame, bbox, (img1.size[1], img1.size[0]))
        emb2 = extract_region_feats(feat_other, bbox, (img1.size[1], img1.size[0]))
        embeddings.append(emb1.squeeze(0).cpu().numpy())
        embeddings.append(emb2.squeeze(0).cpu().numpy())
        labels.extend([0, 1])  # 0: frame_id crop, 1: other_frame_id crop


    np.save(os.path.join(save_path, 'embeddings.npy'), np.array(embeddings))
    np.save(os.path.join(save_path, 'labels.npy'), np.array(labels))
    print("Embeddings saved ✅")



def project_orthogonal_add(a, b):
    proj = torch.sum(a * b, dim=1, keepdim=True) / (torch.norm(b, dim=1, keepdim=True) ** 2 + 1e-8)
    return a + proj * b
def cosine_loss(a, b):
    return 1 - F.cosine_similarity(a, b, dim=1).mean()

def compute_loss(embed_0, embed_1, k=16, head=None, classifier=None):

    B, D = embed_0.shape
    bce_loss = torch.nn.BCEWithLogitsLoss()

    clean_embed = project_orthogonal(embed_0, embed_1)  # (B, D)


    sim_matrix = F.cosine_similarity(embed_1.unsqueeze(1), embed_1.unsqueeze(0), dim=-1)  # (B, B)
    mask = torch.eye(B, device=embed_0.device).bool()
    sim_matrix.masked_fill_(mask, float('inf'))  # 避免选择自己

    _, indices = torch.topk(sim_matrix, k=k, largest=False, dim=1)  # 每行找最不相似的K个背景 index

    all_noisy_embeds = []
    for i in range(B):
        c_i = clean_embed[i].unsqueeze(0)           # (1, D)
        distractors = embed_1[indices[i]]           # (K, D)


        noisy_embeds_i = project_orthogonal_add(c_i.repeat(k, 1), distractors)  # (K, D)

        all_noisy_embeds.append(noisy_embeds_i)

    noisy_embeds = torch.stack(all_noisy_embeds, dim=0)  # (B, K, D)
    noisy_prd = head(noisy_embeds.view(B * k, D))


    loss_noise = cosine_loss(
        noisy_prd,
        clean_embed.unsqueeze(1).expand(B, k, D).reshape(B * k, D)
    )

    org_prd = head(embed_0)
    loss_raw = cosine_loss(org_prd, clean_embed)  # (B, D) 对 (B, D)

    noisy_t = classifier(noisy_prd)
    org_t = classifier(org_prd)
    sun_t = torch.cat((noisy_t,org_t), dim=0)
    label_cheat = torch.zeros_like(sun_t)
    loss_cheat = bce_loss(sun_t, label_cheat)

    loss = loss_noise + 5*loss_raw + loss_cheat
    # loss.backward()
    return loss

class EmbeddingHead(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            # nn.ReLU(),
            nn.Linear(256,256),
            # nn.ReLU(),
            nn.Linear(256, 256),
            # nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        return self.net(x)

def train_head_with_noise_add(embedding_path, label_path, classifier_path,
                                            save_path='addniose2head.pt',
                                            batch_size=64, epochs=10, lr=1e-5):
    dataset = ClipEmbeddingFusionDataset(embedding_path, label_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    classifier = BackgroundClassifier().cuda()
    classifier.load_state_dict(torch.load(classifier_path))
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    head = EmbeddingHead().cuda()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    itr = 0
    for zzz in range(100):
        for epoch in range(epochs):
            head.train()
            total_loss = 0
            for embed_0, embed_1 in dataloader:
                embed_0 = F.normalize(embed_0, dim=-1)
                embed_1 = F.normalize(embed_1, dim=-1)
                embed_0 = embed_0.cuda()
                embed_1 = embed_1.cuda()
                loss = compute_loss(embed_0, embed_1, head=head, classifier=classifier)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * embed_0.size(0)
                itr += 1
                if itr % 500 == 0:
                    print(f"[Itr {itr}] train Loss: {total_loss:.4f} epoch: {epoch}")
                    total_loss = 0
                    torch.save(head.state_dict(), save_path.split('.')[0] + str(itr) + '.pt')
        torch.save(head.state_dict(), save_path)
        return head

'''final work edition'''
img_root = 'a'
label_path ='b'
npy_path ='c'
save_path ='d'
extract_embedding(img_root=img_root, label_path=label_path, npy_path=npy_path, save_path=save_path)
train_classifier(save_path+'embeddings.npy', save_path+'u2mot_labels.npy', save_path=save_path)
train_head_with_noise_add(save_path+'embeddings.npy', save_path+'u2mot_labels.npy', classifier_path=save_path+'backdoor_classifier.pt', save_path=save_path+'/addniose2head.pt')


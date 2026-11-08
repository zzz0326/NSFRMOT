import torch
import torch.nn.functional as F
import open_clip


txt_path = ""
with open(txt_path, "r") as f:
    content = f.read().strip()

car_list = [item.strip() for item in content.split(",") if item.strip() != ""]

print(f"Loaded {len(car_list)} categories.")


prompts = [f"a photo of {obj}" for obj in car_list]


device = "cuda"

model = open_clip.create_model('ViT-L-14', pretrained=None, device=device)

state_dict = torch.load(
    "",
    map_location=device
)

model.load_state_dict(state_dict)
model.eval()

tokenizer = open_clip.get_tokenizer('ViT-L-14')
tokenized = tokenizer(prompts).to(device)


with torch.no_grad():
    text_embs = model.encode_text(tokenized)


text_embs = F.normalize(text_embs, dim=-1)


sim_matrix = text_embs @ text_embs.T  # shape: (N, N)


N = sim_matrix.shape[0]
sim_matrix.fill_diagonal_(0.0)


avg_sim = sim_matrix.mean(dim=1)  # (N,)


k = min(20, N)
values, indices = torch.topk(avg_sim, k=k, largest=False)

selected_embeddings = text_embs[indices]
selected_categories = [car_list[i] for i in indices.tolist()]

print("\nMost dissimilar 20 categories:")
for i, cat in enumerate(selected_categories):
    # print(f"{i+1}: {cat} (avg sim = {values[i].item():.4f})")
    print("'"+str(cat)+"'")
selected_embeddings = F.normalize(selected_embeddings, dim=-1)


sub_sim_matrix = selected_embeddings @ selected_embeddings.T  # (K, K)


K = sub_sim_matrix.shape[0]
mask = ~torch.eye(K, dtype=torch.bool, device=sub_sim_matrix.device)

valid_sims = sub_sim_matrix[mask]

max_sim = valid_sims.max().item()
min_sim = valid_sims.min().item()

print("\nAmong selected embeddings:")
print(f"Max cosine similarity: {max_sim:.6f}")
print(f"Min cosine similarity: {min_sim:.6f}")
# import clip
import torch
from PIL import Image
import os
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import faiss
from collections import defaultdict
import json

from config_run import *  # phải có anno_valid trỏ tới file json của bạn
from transformers import AutoProcessor, AutoModelForVision2Seq   # << đổi ở đây

# ======================
# Utils
# ======================
pwd = os.getcwd()
root = os.path.join(pwd, 'Datasets')
img_folder_vg = os.path.join(root, 'MSCOCO/mscoco/')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_set():
    with open(anno_valid, 'r') as f:
        data = json.load(f)
    rev_id, Go, Ge = [], [], []
    for item in data:
        rev_id.append(os.path.join(img_folder_vg, item['rev']['image_id']))  # gallery image ids
        Go.append(os.path.join(img_folder_vg, item['qe']['image_id']))       # query image ids
        Ge.append(item['qe']['trip'])                                        # (không dùng trong image-only)
    return rev_id, Go, Ge

# ======================
# Qwen2-VL image encoder
# ======================
def _forward_vision(model, pixel_values):
    """
    Tự động tìm vision backbone + projection của Qwen2-VL.
    Trả về tensor [B, D] đã qua projection nếu có, nếu không trả về CLS [B, H].
    """
    # 1) Vision backbone
    if hasattr(model, "vision_model"):
        vis_out = model.vision_model(pixel_values)
        last = vis_out.last_hidden_state
    elif hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        vt = model.model.vision_tower
        vis_out = vt(pixel_values)
        last = vis_out.last_hidden_state if hasattr(vis_out, "last_hidden_state") else vis_out
    else:
        raise RuntimeError("Không tìm thấy vision backbone trong model (vision_model / model.vision_tower).")

    cls = last[:, 0]  # CLS

    # 2) Projection (nếu có)
    proj = None
    for cand in ["vision_proj", "visual_projection", "vision_projection", "proj"]:
        if hasattr(model, cand):
            proj = getattr(model, cand)
            break
        if hasattr(model, "model") and hasattr(model.model, cand):
            proj = getattr(model.model, cand)
            break

    if proj is not None:
        return proj(cls)   # [B, D]
    else:
        return cls         # [B, H] (chưa proj)

@torch.no_grad()
def qwen2vl_image_emb(model, processor, images, device="cuda", batch_size=32, fp16=True):
    """
    images: list[PIL.Image]
    return: torch.Tensor [N, D] (L2-normalized)
    """
    model.eval()
    embs = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.autocast("cuda", enabled=(fp16 and device.startswith("cuda"))):
            z = _forward_vision(model, inputs["pixel_values"])  # [B, D or H]
        z = F.normalize(z, dim=-1)
        embs.append(z.detach().cpu())
    return torch.cat(embs, dim=0)  # [N, D]

# ======================
# Pipeline
# ======================
def create_gallery(model, data_db, processor, device):
    """
    Encode toàn bộ gallery thành 1 tensor [N, D]
    """
    imgs = []
    for p in data_db:
        img = Image.open(p).convert("RGB")
        imgs.append(img)
    feats = qwen2vl_image_emb(model, processor, imgs, device=device, batch_size=32, fp16=True)
    # đóng file để tránh leak
    for im in imgs:
        if hasattr(im, "close"):
            im.close()
    return feats  # [N, D]

def faiss_retrieval_controller(z_que, set_z_rev, images_id_rev, k=50):
    """
    z_que: torch.Tensor [M, D] (đã L2-normalize)
    set_z_rev: torch.Tensor [N, D] (đã L2-normalize)
    images_id_rev: list[str] length N
    """
    if isinstance(z_que, torch.Tensor):
        z_que = z_que.detach().cpu().numpy().astype('float32')
    if isinstance(set_z_rev, torch.Tensor):
        set_z_rev = set_z_rev.detach().cpu().numpy().astype('float32')

    index = faiss.IndexFlatIP(set_z_rev.shape[1])  # inner product ~ cosine (đã normalize)
    index.add(set_z_rev)
    D, I = index.search(z_que, k=k)
    selected_images = [images_id_rev[i] for i in I[0]]
    return selected_images

@torch.no_grad()
def compute_recall_image_only(model, processor, rev_id, image_rev, Go, device, K=[10,20,50]):
    """
    Image-only retrieval: Query là ảnh (Go), Gallery là rev_id (image_rev embeddings)
    """
    hits = defaultdict(int)
    total = len(Go)

    # encode tất cả query ảnh trước cho nhanh
    q_imgs = []
    for p in Go:
        q_imgs.append(Image.open(p).convert("RGB"))
    Q = qwen2vl_image_emb(model, processor, q_imgs, device=device, batch_size=32, fp16=True)  # [M, D]
    for im in q_imgs:
        if hasattr(im, "close"): im.close()

    # search từng query
    for idx, (q_path, q_emb) in enumerate(zip(Go, Q)):
        q_emb = q_emb.unsqueeze(0)  # [1, D]
        topk = faiss_retrieval_controller(q_emb, image_rev, rev_id)  # list các path top-k

        for k in K:
            if q_path in topk[:k]:
                hits[k] += 1

    recall = {k: hits[k] / total for k in K}
    print("========== Recall (Image-only, Qwen2-VL) ==========")
    print(f"R@10: {recall[10]:.5f} | R@20: {recall[20]:.5f} | R@50: {recall[50]:.5f}")
    return recall

# ======================
# Main
# ======================
if __name__ == "__main__":
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load ids
    rev_id, Go, Ge = get_set()  # Ge không dùng trong image-only

    # Qwen2-VL (Instruct hay Base đều dùng được cho image-only)
    MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"  # đổi sang bản nhỏ hơn nếu RAM/GPU hạn chế

    # Processor nhanh (nếu muốn chậm để tái lập đúng cũ: AutoProcessor.from_pretrained(..., use_fast=False))
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # ❗ Dùng đúng loại model cho Vision-Language
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if device == "cuda" else torch.float32,  # dùng `dtype`, tránh warning
        device_map="auto"
    ).eval()

    # Build gallery
    image_rev = create_gallery(model, rev_id, processor, device)  # [N, D]
    print("Gallery size:", image_rev.shape[0])
    print("Vector dim:", image_rev.shape[1])

    # Compute recall cho image-only
    compute_recall_image_only(model, processor, rev_id, image_rev, Go, device)

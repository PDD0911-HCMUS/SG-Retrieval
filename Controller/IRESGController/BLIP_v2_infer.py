# import clip
import math
import torch
from PIL import Image
import open_clip
from config_run import *
import os
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import faiss
from collections import defaultdict
import json
from transformers import AutoTokenizer, Blip2Model, Blip2Processor, AutoProcessor


pwd = os.getcwd()
root = os.path.join(pwd,'Datasets')
img_folder_vg = os.path.join(root,'VisualGenome/VG_100K/')

def get_set():
    with open(anno_valid, 'r') as f:
        data = json.load(f)

    rev_id, Go, Ge = [], [], []
    for item in data:
        rev_id.append(os.path.join(img_folder_vg, item['rev']['image_id']))
        Go.append(item['qe']['image_id'])
        Ge.append(item['qe']['trip'])
        # break
    return rev_id, Go, Ge

def create_gallery(model, data_db, preprocess, device):
    imgs_b = []
    with torch.no_grad():
        for image_id_b in tqdm(data_db):
            
            im = preprocess(images=Image.open(image_id_b).convert('RGB'), return_tensors="pt").to(device)
            z_i = model.get_image_features(**im)
            imgs_b.append(z_i.pooler_output.squeeze(0))

            # break 
        return imgs_b
    
def faiss_retrieval_controller(z_que, set_z_rev, images_id_rev):
    z_que = F.normalize(z_que, p=2, dim=1)
    if isinstance(z_que, torch.Tensor):
        z_que = z_que.detach().cpu().numpy().astype('float32')
    # set_z_rev = np.stack([
    #     t.detach().cpu().numpy() for t in set_z_rev
    # ]).astype('float32')
    set_z_rev = torch.stack(set_z_rev).cpu().numpy().astype('float32')
    index = faiss.IndexFlatIP(set_z_rev.shape[1])  # Dùng Euclidean distance
    index.add(set_z_rev)
    D, I = index.search(z_que, k=50)
    selected_images = [images_id_rev[i].split('/')[-1] for i in I[0]]
    return selected_images

def ndcg_at_k(ranked_ids, pos_set, Ks=(10, 20, 50)):
    """
    ranked_ids: list[str]  — danh sách ID ảnh đã xếp hạng (ví dụ top-50)
    pos_set:    set[str]   — tập các ground-truth (có thể nhiều phần tử)
    Ks:         iterable   — các K cần tính (10,20,50)

    Trả về: dict {f"nDCG@{K}": value}
    """
    # relevance nhị phân (1 nếu ảnh thuộc ground-truth, ngược lại 0)
    rel = [1 if rid in pos_set else 0 for rid in ranked_ids]

    out = {}
    m = len(pos_set)
    for K in Ks:
        rK = rel[:K]
        # DCG@K
        dcg = 0.0
        for i, r in enumerate(rK, start=1):
            if r:
                dcg += 1.0 / math.log2(i + 1)

        # IDCG@K (trường hợp lý tưởng: tất cả positives đứng đầu)
        ideal_hits = min(m, K)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        out[f"nDCG@{K}"] = ndcg
    return out

def ndcg_at_k(ranked_ids, pos_set, Ks=(10, 20, 50)):
    """
    ranked_ids: list[str]  — danh sách ID ảnh đã xếp hạng (ví dụ top-50)
    pos_set:    set[str]   — tập các ground-truth (có thể nhiều phần tử)
    Ks:         iterable   — các K cần tính (10,20,50)

    Trả về: dict {f"nDCG@{K}": value}
    """
    # relevance nhị phân (1 nếu ảnh thuộc ground-truth, ngược lại 0)
    rel = [1 if rid in pos_set else 0 for rid in ranked_ids]

    out = {}
    m = len(pos_set)
    for K in Ks:
        rK = rel[:K]
        # DCG@K
        dcg = 0.0
        for i, r in enumerate(rK, start=1):
            if r:
                dcg += 1.0 / math.log2(i + 1)

        # IDCG@K (trường hợp lý tưởng: tất cả positives đứng đầu)
        ideal_hits = min(m, K)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        out[f"nDCG@{K}"] = ndcg
    return out
    
def compute_recall(model, tokenizer, preprocess, rev_id, image_rev, Go, Ge, device, K = [10, 20, 50]):

    hits_o = defaultdict(int)

    with torch.no_grad():
        for r_id, go, ge in tqdm(zip(rev_id, Go, Ge)):

            inputs = preprocess(images=Image.open(os.path.join(img_folder_vg, go)).convert('RGB'), return_tensors="pt").to(device)
            z = model.get_image_features(**inputs)
            z = z.last_hidden_state
            z = z.mean(dim=1)
            # print(z)
            revO = faiss_retrieval_controller(z, image_rev, rev_id)
            # print(image_id_a[0], image_id_b[0])
            # print(z.size())
            for k in K:
                if(r_id in revO[:k]):
                    hits_o[k] += 1

            # break

        recall_o = {k: hits_o[k] / len(rev_id) for k in K}

        print(f"========== Recall for Cross ==========")
        print(f"only Image | R@10: {recall_o[10]:.5f} | R@20: {recall_o[20]:.5f} | R@50: {recall_o[50]:.5f}")

    return 

def compute_ndcg_only_images(model, preprocess, rev_id, image_rev, Go, Ge, device, tgt_lst, K = [10, 20, 50]):

    sum_ndcg = defaultdict(float)
    n_query = 0

    with torch.no_grad():
        for r_id, go, ge in tqdm(zip(rev_id, Go, Ge)):
            image = Image.open(os.path.join(img_folder_vg, go)).convert('RGB')
            inputs = preprocess(images=image, return_tensors="pt").to(device)
            z = model.get_image_features(**inputs)
            z = z.pooler_output
            revO = faiss_retrieval_controller(z, image_rev, rev_id)
            tgt = get_tgt_by_image(go, tgt_lst)
            pos_set = set(tgt)

            # tính nDCG@K cho query này
            q_ndcg = ndcg_at_k(revO, pos_set, Ks=K)
            for key, val in q_ndcg.items():
                sum_ndcg[key] += val

            n_query += 1

            # break

    # trung bình trên tất cả query
    mean_ndcg = {key: (sum_ndcg[key] / max(1, n_query)) for key in sum_ndcg}

    print("========== nDCG (only Images) ==========")
    # in theo thứ tự K
    for k in K:
        print(f"nDCG@{k}: {mean_ndcg.get(f'nDCG@{k}', 0.0):.5f}")

    return mean_ndcg

def get_tgt_by_image(image_id, tgt_lst):
    for item in tgt_lst:
        if(item['image_query'] == image_id):

            return item['target']
    
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rev_id, Go, Ge = get_set()

    tgt_pth = '/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/VisualGenome/Target.json'
    with open(tgt_pth) as f:
        tgt_lst = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

    image_rev = create_gallery(model, rev_id, processor, device)
    print("Gallery size:", len(image_rev))
    print("Sample vector shape:", image_rev[0].shape)
    print("All vectors same shape:", all(t.shape == image_rev[0].shape for t in image_rev))

    # compute_recall(model, tokenizer, processor, rev_id, image_rev, Go, Ge, device)

    compute_ndcg_only_images(model, processor, rev_id, image_rev, Go, Ge, device, tgt_lst)






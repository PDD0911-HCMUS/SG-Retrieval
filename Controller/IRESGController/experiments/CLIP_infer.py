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

pwd = os.getcwd()
root = os.path.join(pwd,'Datasets')
img_folder_vg = os.path.join(root,'MSCOCO/mscoco/')

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

            im = preprocess(Image.open(image_id_b).convert('RGB')).unsqueeze(0).to(device)

            z_i = model.encode_image(im)
            
            z_i = F.normalize(z_i, p=2, dim=1)

            imgs_b.append(z_i.squeeze(0))

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
    
def compute_recall_only_images(model, rev_id, image_rev, Go, Ge, device, pre_trained, K = [10, 20, 50]):

    hits_o = defaultdict(int)

    with torch.no_grad():
        for r_id, go, ge in tqdm(zip(rev_id, Go, Ge)):

            inputs = preprocess(Image.open(os.path.join(img_folder_vg, go)).convert('RGB')).unsqueeze(0).to(device)
            z = model.encode_image(inputs)
            revO = faiss_retrieval_controller(z, image_rev, rev_id)
            for k in K:
                if(r_id in revO[:k]):
                    hits_o[k] += 1

            # # break

        recall_o = {k: hits_o[k] / len(rev_id) for k in K}
        print(f"========== Recall only Images {pre_trained} ==========")
        print(f"non-Editted | R@10: {recall_o[10]:.5f} | R@20: {recall_o[20]:.5f} | R@50: {recall_o[50]:.5f}")

    return 

def compute_recall_only_graph(model, preprocess, rev_id, image_rev, Go, Ge, device, pre_trained, K = [10, 20, 50]):

    hits_o = defaultdict(int)

    with torch.no_grad():
        for r_id, go, ge in tqdm(zip(rev_id, Go, Ge)):

            inputs = preprocess(Image.open(os.path.join(img_folder_vg, go)).convert('RGB')).unsqueeze(0).to(device)
            z = model.encode_image(inputs)
            revO = faiss_retrieval_controller(z, image_rev, rev_id)
            
            for k in K:
                if(r_id in revO[:k]):
                    hits_o[k] += 1

            break

        recall_o = {k: hits_o[k] / len(rev_id) for k in K}
        print(f"========== Recall only Graphs {pre_trained} ==========")
        print(f"non-Editted | R@10: {recall_o[10]:.5f} | R@20: {recall_o[20]:.5f} | R@50: {recall_o[50]:.5f}")

    return 

def compute_recall_cross(model, tokenizer, preprocess, rev_id, image_rev, Go, Ge, device, pre_trained, K = [10, 20, 50]):

    hits_o = defaultdict(int)

    with torch.no_grad():
        for r_id, go, ge in tqdm(zip(rev_id, Go, Ge)):
            text = ge
            inputs_im = preprocess(Image.open(os.path.join(img_folder_vg, go)).convert('RGB')).unsqueeze(0).to(device)
            inputs_txt = tokenizer(text).to(device) 
            z_i = model.encode_image(inputs_im)
            z_t = model.encode_text(inputs_txt)
            z_t = z_t.mean(dim=0).unsqueeze(0)
            z = z_i + z_t
            revO = faiss_retrieval_controller(z, image_rev, rev_id)
            # print(revO)
            r_id = r_id.split('/')[-1]
            for k in K:
                if(r_id in revO[:k]):
                    hits_o[k] += 1

            # break

        recall_o = {k: hits_o[k] / len(rev_id) for k in K}
        print(f"========== Recall only Cross {pre_trained} ==========")
        print(f"non-Editted | R@10: {recall_o[10]:.5f} | R@20: {recall_o[20]:.5f} | R@50: {recall_o[50]:.5f}")

    return 

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

def compute_ndcg_only_images(model, preprocess, rev_id, image_rev, Go, Ge, device, tgt_lst, pre_trained, K = [10, 20, 50]):

    sum_ndcg = defaultdict(float)
    n_query = 0
    
    with torch.no_grad():
        for r_id, go, ge in tqdm(zip(rev_id, Go, Ge)):
            inputs = preprocess(Image.open(os.path.join(img_folder_vg, go)).convert('RGB')).unsqueeze(0).to(device)
            z = model.encode_image(inputs)
            revO = faiss_retrieval_controller(z, image_rev, rev_id)

            tgt = get_tgt_by_image(go, tgt_lst)

            # print(go)
            # print(tgt)
            pos_set = set(tgt)

            # tính nDCG@K cho query này
            q_ndcg = ndcg_at_k(revO, pos_set, Ks=K)
            for key, val in q_ndcg.items():
                sum_ndcg[key] += val

            n_query += 1

            # break

    # trung bình trên tất cả query
    mean_ndcg = {key: (sum_ndcg[key] / max(1, n_query)) for key in sum_ndcg}

    print(f"========== nDCG only Images {pre_trained} ==========")
    # in theo thứ tự K
    for k in K:
        print(f"nDCG@{k}: {mean_ndcg.get(f'nDCG@{k}', 0.0):.5f}")

    return mean_ndcg

def compute_ndcg_only_graph(model, preprocess, rev_id, image_rev, Go, Ge, device, tgt_lst, pre_trained, K = [10, 20, 50]):

    sum_ndcg = defaultdict(float)
    n_query = 0

    with torch.no_grad():
        for r_id, go, ge in tqdm(zip(rev_id, Go, Ge)):
            text = ge
            inputs = preprocess(text).to(device) 
            z = model.encode_text(inputs)
            z = z.mean(dim=0).unsqueeze(0)
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

    print(f"========== nDCG only Graphs {pre_trained} ==========")
    # in theo thứ tự K
    for k in K:
        print(f"nDCG@{k}: {mean_ndcg.get(f'nDCG@{k}', 0.0):.5f}")

    return mean_ndcg

def compute_ndcg_cross(model, tokenizer, preprocess, rev_id, image_rev, Go, Ge, device, tgt_lst, pre_trained, K = [10, 20, 50]):

    sum_ndcg = defaultdict(float)
    n_query = 0

    with torch.no_grad():
        for r_id, go, ge in tqdm(zip(rev_id, Go, Ge)):
            text = ge
            inputs_im = preprocess(Image.open(os.path.join(img_folder_vg, go)).convert('RGB')).unsqueeze(0).to(device)
            inputs_txt = tokenizer(text).to(device) 
            z_i = model.encode_image(inputs_im)
            z_t = model.encode_text(inputs_txt)
            z_t = z_t.mean(dim=0).unsqueeze(0)
            z = z_i + z_t
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

    print(f"========== nDCG only Cross {pre_trained} ==========")
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

    pre_trained = 'ViT-L-14'

    rev_id, Go, Ge = get_set()

    tgt_pth = '/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/MSCOCO/Target_mscoco.json'
    with open(tgt_pth) as f:
        tgt_lst = json.load(f)

    model, _, preprocess = open_clip.create_model_and_transforms(pre_trained, pretrained='openai')
    tokenizer = open_clip.get_tokenizer(pre_trained)

    image_rev = create_gallery(model, rev_id, preprocess, device)
    print("Gallery size:", len(image_rev))
    print("Sample vector shape:", image_rev[0].shape)
    print("All vectors same shape:", all(t.shape == image_rev[0].shape for t in image_rev))

    # compute_recall_only_images(model, rev_id, image_rev, Go, Ge, device)
    # compute_recall_cross(model, tokenizer, preprocess, rev_id, image_rev, Go, Ge, device, pre_trained)

    compute_ndcg_only_images(model, preprocess, rev_id, image_rev, Go, Ge, device, tgt_lst, pre_trained)
    compute_ndcg_only_graph(model, tokenizer, rev_id, image_rev, Go, Ge, device, tgt_lst, pre_trained)
    compute_ndcg_cross(model, tokenizer, preprocess, rev_id, image_rev, Go, Ge, device, tgt_lst, pre_trained)



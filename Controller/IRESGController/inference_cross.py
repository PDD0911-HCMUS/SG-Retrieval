import math
import torch
from typing import Iterable
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from Controller.IRESGController.model.model_v2 import build, ModelCross
from tqdm import tqdm
import faiss
from collections import defaultdict
import torch.nn.functional as F
import json
import os
from config_run import *
from Controller.IRESGController.dataset.create_db import create_db, collate_fn_dual_image_db

def get_model():
    model, _ = build(hidden_dim,lr_backbone,masks, backbone, dilation, 
                nhead, nlayer, d_ffn, dropout, random_erasing_prob, activation, pre_train)
    
    checkpoint = torch.load(ckpt, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to(device)
    return model


def faiss_retrieval_controller(z_que, set_z_rev, images_id_rev):
    z_que = F.normalize(z_que, p=2, dim=1)
    if isinstance(z_que, torch.Tensor):
        z_que = z_que.detach().cpu().numpy().astype('float32')
    set_z_rev = np.stack([
        t.detach().cpu().numpy() for t in set_z_rev
    ]).astype('float32')
    index = faiss.IndexFlatIP(set_z_rev.shape[1])  # Dùng Euclidean distance
    index.add(set_z_rev)
    D, I = index.search(z_que, k=50)
    selected_images = [images_id_rev[i] for i in I[0]]
    return selected_images

def get_embedding(model: ModelCross, img_id, img, triplet, mode, device):
    with torch.no_grad():
        img = img[0].to(device)
        triplet = [{k: v.to(device) for k, v in t.items()} for t in triplet]

        z_i, z_i_msk, _ = model.models.vision_encoder(img)

        if(mode == 0):
            go, _ = model.models.graph_encoder_o(triplet) 
            z_cross, _ = model.models.attn_graph_o(
                query=go,
                key=z_i,
                value=z_i,
                key_padding_mask=z_i_msk
            )
            z_cross = F.normalize(z_cross, p=2, dim=1)
        if(mode == 1):
            ge, _ = model.models.graph_encoder_e(triplet)
            z_cross, _ = model.models.attn_graph_be(
                query=ge,
                key=z_i,
                value=z_i,
                key_padding_mask=z_i_msk
            )
            z_cross = F.normalize(z_cross, p=2, dim=1)

        return img_id[0], z_cross[:,0][0]

def create_gallery(model: ModelCross, data_db: Iterable, device):
    images_ids_b = []
    images_rev = []
    # imgs_b = []
    with torch.no_grad():
        for img_a, img_b, trip_que, trip_rev, image_id_a, image_id_b in tqdm(data_db):

            images_ids_b.append(image_id_b[0])

            img_b = img_b[0].to(device)

            z_iB, z_iB_msk, _ = model.models.vision_encoder(img_b)
            
            z_iB = model.models.proj(z_iB[:,0])

            z_iB = F.normalize(z_iB, p=2, dim=1)

            images_rev.append(z_iB[0])

        return images_ids_b, images_rev
    
def get_set(json_file):
    # que_id, rev_id, Go, Ge = [], [], [], []
    image_ids, triplets = [], []
    if(len(json_file) > 1):
        for file in json_file:
            with open(file, 'r') as f:
                data = json.load(f)

            for item in tqdm(data):
                image_ids.append(os.path.join(vg_image_dir, item['rev']['image_id']))
                image_ids.append(os.path.join(vg_image_dir, item['qe']['image_id']))
                triplets.append(item['qe']['trip'])
                triplets.append(item['rev']['trip'])

    return image_ids, triplets

def compute_top_accuracy(model: ModelCross, data_db: Iterable, device, K = [10, 20, 50]):

    image_ids_a = []

    print(f"Creating Gallery")
    images_ids_b, images_rev = create_gallery(model, data_db, device)

    hits_o = defaultdict(int)

    print(f"Start Running Validation")
    with torch.no_grad():
        for img_a, img_b, trip_que, trip_rev, image_id_a, image_id_b in tqdm(data_db):
            image_ids_a.append(image_id_a[0])

            img_a = img_a[0].to(device)
            trip_que = [{k: v.to(device) for k, v in t.items()} for t in trip_que]

            z_iA, z_iA_msk, _ = model.models.vision_encoder(img_a)
            go, _ = model.models.graph_encoder_o(trip_que)

            z_o, _ = model.models.attn_graph_o(
                query=go,
                key=z_iA,
                value=z_iA,
                key_padding_mask=z_iA_msk)
            z_o = model.models.proj(z_o[:, 0])

            revO = faiss_retrieval_controller(z_o, images_rev, images_ids_b)
            for k in K:
                if(image_id_b[0] in revO[:k]):
                    hits_o[k] += 1

            # break

        Acc_o = {k: hits_o[k] / len(data_db) for k in K}
        # print("Recall@K for z_o:", recall_o)
        # print("Recall@K for z_e:", recall_e)
        print(f"========== Recall for only Images ==========")
        print(f"Images | Acc@10: {Acc_o[10]:.5f} | Acc@20: {Acc_o[20]:.5f} | Acc@50: {Acc_o[50]:.5f}")

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

def compute_ndcg(model: ModelCross, data_db, device, K = [10, 20, 50]):
    """
    Giống compute_recall nhưng tính nDCG@K.
    YÊU CẦU: faiss_retrieval_controller(z, images_rev, images_ids_b) trả về list ID ảnh đã xếp hạng (ví dụ n=50).
    Ground-truth hiện tại: mỗi record có 1 ảnh đúng (image_id_b[0]).
    Nếu sau này bạn có multi-GT, chỉ cần thay pos_set lại cho phù hợp.
    """
    print("Creating Gallery")
    images_ids_b, images_rev = create_gallery(model, data_db, device)

    sum_ndcg = defaultdict(float)
    n_query = 0
    tgt_pth = '/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/MSCOCO/Target_mscoco.json'
    with open(tgt_pth) as f:
        tgt_lst = json.load(f)

    print("Start Running Validation (nDCG)")
    with torch.no_grad():
        for img_a, img_b, trip_que, trip_rev, image_id_a, image_id_b in tqdm(data_db):


            img_a = img_a[0].to(device)
            trip_que = [{k: v.to(device) for k, v in t.items()} for t in trip_que]

            z_iA, z_iA_msk, _ = model.models.vision_encoder(img_a)
            go, _ = model.models.graph_encoder_o(trip_que)

            z_o, _ = model.models.attn_graph_o(
                query=go,
                key=z_iA,
                value=z_iA,
                key_padding_mask=z_iA_msk)
            z_o = model.models.proj(z_o[:, 0])

            ranked_ids = faiss_retrieval_controller(z_o, images_rev, images_ids_b)

            # ---- ground-truth ----
            # hiện tại GT là 1 ảnh: image_id_b[0]
            # nếu sau này có nhiều GT: pos_set = set(list_ground_truth_ids)

            # print(image_id_a[0])

            tgt = get_tgt_by_image(image_id_a[0], tgt_lst)
            pos_set = set(tgt)

            # tính nDCG@K cho query này
            q_ndcg = ndcg_at_k(ranked_ids, pos_set, Ks=K)
            for key, val in q_ndcg.items():
                sum_ndcg[key] += val

            n_query += 1

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
    dataset_db = create_db(
        image_folder=vg_image_dir,
        ann_file=anno_valid,
        tokenizer=tokenizer,
        max_length=max_lenght
    )

    sampler_db = SequentialSampler(dataset_db)
    data_db = DataLoader(dataset_db,
            batch_size=1, 
            sampler=sampler_db,
            drop_last=False,
            collate_fn=collate_fn_dual_image_db,
            num_workers=num_workers,
            pin_memory=True)
    
    model = get_model()
    
    checkpoint = torch.load(ckpt, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to(device)
    # create_gallery(model, data_db, device)
    # compute_top_accuracy(model, data_db, device)
    compute_ndcg(model, data_db, device)
    pass
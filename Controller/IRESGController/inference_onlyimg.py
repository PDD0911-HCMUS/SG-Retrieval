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

            z_iA, z_iA_msk, _ = model.models.vision_encoder(img_a)
            z_iA = model.models.proj(z_iA[:,0])

            revO = faiss_retrieval_controller(z_iA, images_rev, images_ids_b)
            for k in K:
                if(image_id_b[0] in revO[:k]):
                    hits_o[k] += 1

            # break

        Acc_o = {k: hits_o[k] / len(data_db) for k in K}
        # print("Recall@K for z_o:", recall_o)
        # print("Recall@K for z_e:", recall_e)
        print(f"========== Recall for only Images ==========")
        print(f"Images | Acc@10: {Acc_o[10]:.5f} | Acc@20: {Acc_o[20]:.5f} | Acc@50: {Acc_o[50]:.5f}")
    
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
    compute_top_accuracy(model, data_db, device)
    pass
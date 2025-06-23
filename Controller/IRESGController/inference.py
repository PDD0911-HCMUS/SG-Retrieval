import torch
from typing import Iterable
import time
import numpy as np
from Controller.IRESGController.model.model import ModelCross
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
from Controller.IRESGController.model.model import build, ModelCross
from tqdm import tqdm
import faiss
from collections import defaultdict
import torch.nn.functional as F
import json
import os
from config_run import *
from Controller.IRESGController.dataset.create_db import create_db, collate_fn_dual_image_db

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

def create_gallery(model: ModelCross, data_db: Iterable, device):
    images_ids = []
    triplets = []
    # imgs_b = []
    with torch.no_grad():
        for img_a, img_b, trip_que, trip_rev, image_id_a, image_id_b in tqdm(data_db):

            images_ids.append(image_id_b[0])
            images_ids.append(image_id_a[0])

            img_b = img_b[0].to(device)
            img_a = img_a[0].to(device)

            trip_rev = [{k: v.to(device) for k, v in t.items()} for t in trip_rev]
            trip_que = [{k: v.to(device) for k, v in t.items()} for t in trip_que]

            z_iB, z_iB_msk, _ = model.models.vision_encoder(img_b)
            z_iA, z_iA_msk, _ = model.models.vision_encoder(img_a)

            ge, _ = model.models.graph_encoder_e(trip_rev)
            go, _ = model.models.graph_encoder_o(trip_que)

            z_o, _ = model.models.attn_graph_o(
                query=go,
                key=z_iA,
                value=z_iA,
                key_padding_mask=z_iA_msk
            )
            
            z_eb, _ = model.models.attn_graph_be(
                query=ge,
                key=z_iB,
                value=z_iB,
                key_padding_mask=z_iB_msk
            )
            # imgs_b.append(z_iB[:,0])

            z_eb = F.normalize(z_eb, p=2, dim=1)
            z_o = F.normalize(z_o, p=2, dim=1)
            triplets.append(z_eb[:,0][0])
            triplets.append(z_o[:,0][0])

        return images_ids, triplets
    
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
    
if __name__ == "__main__":
    dataset_db = create_db(
        image_folder=vg_image_dir,
        ann_file=anno,
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
    
    model, _ = build(hidden_dim,lr_backbone,masks, backbone, dilation, 
                nhead, nlayer, d_ffn, dropout, random_erasing_prob, activation, pre_train)
    
    checkpoint = torch.load(ckpt, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    create_gallery(model, data_db, device)
    pass
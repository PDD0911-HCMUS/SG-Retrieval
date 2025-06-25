import torch
from typing import Iterable
import numpy as np
from Controller.IRESGController.model.model import ModelCross
from torch.utils.data import DataLoader, SequentialSampler
from Controller.IRESGController.model.model import build, ModelCross
from tqdm import tqdm
import faiss
from collections import defaultdict
import torch.nn.functional as F
import json
import os
from config_run import *
from Controller.IRESGController.dataset.create_db import create_db, collate_fn_dual_image_db
import Entities.entities as entity
from sqlalchemy.exc import SQLAlchemyError
from config import db

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
        IRESGVG = entity.IRESGVG
        # try:
            
        for img_a, img_b, trip_que, trip_rev, image_id_a, image_id_b in tqdm(data_db):

            im_id_o, z_cross_o = get_embedding(model, image_id_a, img_a, trip_que, 0, device)
            im_id_e, z_cross_e = get_embedding(model, image_id_b, img_b, trip_rev, 1, device)

            insert_o = IRESGVG(
                image_id = im_id_o,
                cross_embedding = z_cross_o
            )

            insert_e = IRESGVG(
                image_id = im_id_e,
                cross_embedding = z_cross_e
            )

            db.session.add(insert_o)
            db.session.add(insert_e)
            break
        db.session.commit()
        # except SQLAlchemyError as e:
        #     print(str(e))
        #     db.session.rollback()
        # finally:
        #     db.session.close()

        return
    
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
    model = model.to(device)
    create_gallery(model, data_db, device)
    pass
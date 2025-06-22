import torch
from typing import Iterable
import time
import numpy as np
from Controller.IRESGController.model.model import ModelCross
from tqdm import tqdm
import faiss
from collections import defaultdict
import torch.nn.functional as F
import json
import os
from config_run import *

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
    images_ids_b = []
    triplets_rev = []
    # imgs_b = []
    with torch.no_grad():
        for img_a, img_b, trip_que, trip_rev, image_id_a, image_id_b in tqdm(data_db):

            images_ids_b.append(image_id_b[0])

            img_b = img_b[0].to(device)
            trip_rev = [{k: v.to(device) for k, v in t.items()} for t in trip_rev]
            z_iB, z_iB_msk, _ = model.models.vision_encoder(img_b)

            ge, _ = model.models.graph_encoder_e(trip_rev)
            
            z_eb, _ = model.models.attn_graph_be(
                query=ge,
                key=z_iB,
                value=z_iB,
                key_padding_mask=z_iB_msk
            )
            # imgs_b.append(z_iB[:,0])
            z_eb = F.normalize(z_eb, p=2, dim=1)
            triplets_rev.append(z_eb[:,0][0])

        return images_ids_b, triplets_rev
    
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
    image_ids, triplets = get_set([anno_train, anno_valid])
    # get_set(anno_valid)
    print(len(image_ids), len(triplets))
    pass
# import clip
import torch
from PIL import Image
import open_clip
from Controller.IRESGController.dataset.create_db import create_db, collate_fn_dual_image_db
from torch.utils.data import DataLoader, SequentialSampler
from config_run import *
import os
import time
import datetime
import random
import numpy as np
import torch.nn.functional as F
from typing import Iterable
from tqdm import tqdm
import faiss
from collections import defaultdict

pwd = os.getcwd()
root = os.path.join(pwd,'Datasets')
img_folder_vg = os.path.join(root,'VisualGenome/VG_100K/')

def set_seed(seed=42):
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch random seed
    torch.cuda.manual_seed(seed)  # Cho GPU
    
def create_gallery(model, data_db, preprocess, device):
    images_ids_b = []
    imgs_b = []
    with torch.no_grad():
        for img_a, img_b, trip_que, trip_rev, image_id_a, image_id_b in tqdm(data_db):

            images_ids_b.append(image_id_b[0])

            im = preprocess(Image.open(os.path.join(img_folder_vg, image_id_b[0])).convert('RGB')).unsqueeze(0).to(device)

            z_i = model.encode_image(im)
            
            z_i = F.normalize(z_i, p=2, dim=1)

            imgs_b.append(z_i)

        return images_ids_b, z_i
    
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
    
def compute_recall(model, data_db: Iterable, device, K = [10, 20, 50]):

    image_ids_a = []
    # triplets_que_o, triplets_que_e = [], []
    imgs_a = []

   
    images_ids_b, triplets_rev = create_gallery(model, data_db, device)

    hits_o = defaultdict(int)
    hits_e = defaultdict(int)

    with torch.no_grad():
        for img_a, img_b, trip_que, trip_rev, image_id_a, image_id_b in tqdm(data_db):

            trip_que = [{k: v.to(device) for k, v in t.items()} for t in trip_que]
            trip_rev = [{k: v.to(device) for k, v in t.items()} for t in trip_rev]

            revO = faiss_retrieval_controller(z_o[:,0][0].unsqueeze(0), triplets_rev, images_ids_b)
            revE = faiss_retrieval_controller(z_e[:,0][0].unsqueeze(0), triplets_rev, images_ids_b)
            # print(image_id_a[0], image_id_b[0])
            for k in K:
                if(image_id_b[0] in revO[:k]):
                    hits_o[k] += 1
                if(image_id_b[0] in revE[:k]):
                    hits_e[k] += 1

            # break

        recall_o = {k: hits_o[k] / len(data_db) for k in K}
        recall_e = {k: hits_e[k] / len(data_db) for k in K}
        # print("Recall@K for z_o:", recall_o)
        # print("Recall@K for z_e:", recall_e)
        print(f"========== Recall for non-Editted and Editted ==========")
        print(f"non-Editted | R@10: {recall_o[10]:.5f} | R@20: {recall_o[20]:.5f} | R@50: {recall_o[50]:.5f}")
        print(f"Editted     | R@10: {recall_e[10]:.5f} | R@20: {recall_e[20]:.5f} | R@50: {recall_e[50]:.5f}")

    return 
    
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_db = create_db(
        image_folder=vg_image_dir,
        ann_file=anno_valid,
        tokenizer=tokenizer,
        max_length=max_lenght
    )

    sampler_val = SequentialSampler(dataset_db)
    data_db = DataLoader(dataset_db,
                batch_size=1, 
                sampler=sampler_val,
                drop_last=False,
                collate_fn=collate_fn_dual_image_db,
                num_workers=num_workers,
                pin_memory=True)

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    create_gallery(model, data_db, preprocess, device)



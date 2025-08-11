# import clip
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

def set_seed(seed=42):
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch random seed
    torch.cuda.manual_seed(seed)  # Cho GPU

def get_set():
    with open(anno_valid, 'r') as f:
        data = json.load(f)

    rev_id, Go, Ge = [], [], []
    for item in data:
        rev_id.append(os.path.join(vg_image_dir, item['rev']['image_id']))
        Go.append(item['qe']['trip'])
        Ge.append(item['rev']['trip'])
    return rev_id, Go, Ge

def create_gallery(model, data_db, preprocess, device):
    imgs_b = []
    with torch.no_grad():
        for image_id_b in tqdm(data_db):

            im = preprocess(Image.open(image_id_b).convert('RGB')).unsqueeze(0).to(device)

            z_i = model.encode_image(im)
            
            z_i = F.normalize(z_i, p=2, dim=1)

            imgs_b.append(z_i.squeeze(0))

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
    selected_images = [images_id_rev[i] for i in I[0]]
    return selected_images
    
def compute_recall(model, rev_id, image_rev, Go, Ge, device, K = [10, 20, 50]):

    hits_o = defaultdict(int)
    hits_e = defaultdict(int)

    with torch.no_grad():
        for r_id, go, ge in tqdm(zip(rev_id, Go, Ge)):

            token_zo = tokenizer(go).to(device) 
            token_ze = tokenizer(ge).to(device) 
            
            zo = model.encode_text(token_zo)
            ze = model.encode_text(token_ze)

            revO = faiss_retrieval_controller(zo, image_rev, rev_id)
            revE = faiss_retrieval_controller(ze, image_rev, rev_id)
            # print(image_id_a[0], image_id_b[0])
            for k in K:
                if(r_id in revO[:k]):
                    hits_o[k] += 1
                if(r_id in revE[:k]):
                    hits_e[k] += 1

            # break

        recall_o = {k: hits_o[k] / len(rev_id) for k in K}
        recall_e = {k: hits_e[k] / len(rev_id) for k in K}
        # print("Recall@K for z_o:", recall_o)
        # print("Recall@K for z_e:", recall_e)
        print(f"========== Recall for non-Editted and Editted ==========")
        print(f"non-Editted | R@10: {recall_o[10]:.5f} | R@20: {recall_o[20]:.5f} | R@50: {recall_o[50]:.5f}")
        print(f"Editted     | R@10: {recall_e[10]:.5f} | R@20: {recall_e[20]:.5f} | R@50: {recall_e[50]:.5f}")

    return 
    
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rev_id, Go, Ge = get_set()

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    image_rev = create_gallery(model, rev_id, preprocess, device)
    print("Gallery size:", len(image_rev))
    print("Sample vector shape:", image_rev[0].shape)
    print("All vectors same shape:", all(t.shape == image_rev[0].shape for t in image_rev))

    compute_recall(model, rev_id, image_rev, Go, Ge, device)




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
from transformers import AutoProcessor, BlipModel, BlipForImageTextRetrieval

pwd = os.getcwd()
root = os.path.join(pwd,'Datasets')
img_folder_vg = os.path.join(root,'VisualGenome/VG_100K/')

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
        rev_id.append(os.path.join(img_folder_vg, item['rev']['image_id']))
        Go.append(item['qe']['image_id'])
        Ge.append(item['qe']['trip'])
    return rev_id, Go, Ge

def create_gallery(model, data_db, preprocess, device):
    imgs_b = []
    with torch.no_grad():
        for image_id_b in tqdm(data_db):
            
            im = preprocess(images=Image.open(image_id_b).convert('RGB'), return_tensors="pt").to(device)
            z_i = model.get_image_features(**im)
            
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
    selected_images = [images_id_rev[i] for i in I[0]]
    return selected_images
    
def compute_recall(model, preprocess, rev_id, image_rev, Go, Ge, device, K = [10, 20, 50]):

    hits_o = defaultdict(int)

    with torch.no_grad():
        for r_id, go, ge in tqdm(zip(rev_id, Go, Ge)):
            # image = Image.open(os.path.join(img_folder_vg, go)).convert('RGB')
            text = ge
            inputs = preprocess(text=text, padding=True, return_tensors="pt").to(device)
            z = model.get_text_features(**inputs)
            # image_emb = z.image_embeds
            # text_emb = z.text_embeds.mean(dim=0).unsqueeze(0)
            # z = image_emb + text_emb
            z = z.mean(dim=0).unsqueeze(0)
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
    
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rev_id, Go, Ge = get_set()

    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device).eval()

    image_rev = create_gallery(model, rev_id, processor, device)
    print("Gallery size:", len(image_rev))
    print("Sample vector shape:", image_rev[0].shape)
    print("All vectors same shape:", all(t.shape == image_rev[0].shape for t in image_rev))

    compute_recall(model, processor, rev_id, image_rev, Go, Ge, device)




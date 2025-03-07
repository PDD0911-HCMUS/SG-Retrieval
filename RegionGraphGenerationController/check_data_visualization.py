import json
from torch.utils.data import Dataset
import torch
import gc
import os
from PIL import Image, ImageDraw
import datasets.transform as T
from transformers import AutoTokenizer


vg_image_dir = "/home/duypd/ThisPC-DuyPC/SG-Retrieval/0_Datasets/VisualGenome/VG_100K/"
vg_anno_train = "/home/duypd/ThisPC-DuyPC/SG-Retrieval/0_Datasets/VisualGenome/anno_rg/train_data.json"
vg_anno_val = "/home/duypd/ThisPC-DuyPC/SG-Retrieval/0_Datasets/VisualGenome/anno_rg/val_data.json"


with open(vg_anno_train, 'r') as f:
    data = json.load(f)

idx = 70

img_id = data[idx]['image_id']
img_path = os.path.join(vg_image_dir,str(img_id) + '.jpg')

img = Image.open(img_path)
target = data[idx]['regions']

img.show()

bbox_rg =  [obj["bbox"] for obj in target]
phrases = [obj["phrase"] for obj in target]

draw = ImageDraw.Draw(img)

for item, region in zip(bbox_rg, phrases):
    x, y, w, h = item[0], item[1], item[2], item[3]
    phrase = region
    draw.rectangle([x, y, x + w, y + h], outline="blue", width=2)

    text_size = draw.textbbox((x, y), phrase)
    text_w, text_h = text_size[2] - text_size[0], text_size[3] - text_size[1]
    draw.rectangle([x, y - text_h - 5, x + text_w + 10, y], fill="black")
    draw.text((x + 5, y - text_h - 5), phrase, fill="white")

img.show()
from Controller.IRESGController.model.model import build
import config as args
import torch
import torchvision.transforms as T
import json
from PIL import Image
import os
from typing import List, Dict
from transformers import BertTokenizer
import torch.nn.functional as F

transform = T.Compose([
    T.Resize(512),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_model():
    #Transformer encoder:
    hidden_dim=256
    nhead=8
    nlayer=6
    d_ffn=2048
    dropout=0.1
    activation="relu"
     
    #Vision Encoder:
    position_embedding='sine'
    backbone='resnet50' # choose resnet50, resnet101, 
    dilation=False
    frozen_weights=None
    lr_backbone=1e-05
    masks=False

    #Graph Encoder:
    random_erasing_prob=0.3
    pre_train = 'bert-base-uncased'

    ckpt = args.Checkpoint.ckpt_IRESG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build(hidden_dim,lr_backbone,masks, backbone, dilation, 
                nhead, nlayer, d_ffn, dropout, random_erasing_prob, activation, pre_train)
    model.load_state_dict(torch.load(ckpt, map_location=device)['model_state_dict'])
    return model

def encode_triplets(triplets: List[str]) -> Dict[str, torch.Tensor]:
    enc = tokenizer(
        triplets,
        padding='max_length',
        truncation=True,
        max_length=7,
        return_tensors='pt'
    )
    return {
        'trip': enc['input_ids'],         # shape: [num_triplets, max_len]
        'trip_msk': enc['attention_mask'] # shape: [num_triplets, max_len]
    }

def pad_or_truncate_tensor(item: Dict[str, torch.Tensor], max_i: int = 10) -> Dict[str, torch.Tensor]:
    for key in ['trip', 'trip_msk']:
        if key in item:
            seq = item[key]
            if seq.size(0) < max_i:
                padding = torch.zeros((max_i - seq.size(0), seq.size(1)), dtype=seq.dtype)
                item[key] = torch.cat([seq, padding], dim=0)
            elif seq.size(0) > max_i:
                item[key] = seq[:max_i]
    return item

def process_batch(tensor_list: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    return [pad_or_truncate_tensor(item) for item in tensor_list]

def create_input(im, trip, model):
    img = transform(im).unsqueeze(0)
    model.eval()
    print(model.eval())
    with torch.no_grad():
        output_a = model.model_a(img, trip)
        output_b = model.model_b(img, trip)

        output_a = F.normalize(output_a, dim=1)
        output_b = F.normalize(output_b, dim=1)
        return output_a, output_b

if __name__ == "__main__":
    model = get_model()
    anno_train = args.ConfigData.iresg_train
    vg_image_dir = args.ConfigData.img_folder_vg

    with open(anno_train, 'r') as f:
        data = json.load(f)

    sample = data[0]['qe']

    image = Image.open(os.path.join(vg_image_dir, sample['image_id'])).convert('RGB')
    trip = sample['trip']
    trip = encode_triplets(trip)
    trip = [pad_or_truncate_tensor(trip)]
    output_a, output_b = create_input(image, trip, model)
    print(output_a, output_b)

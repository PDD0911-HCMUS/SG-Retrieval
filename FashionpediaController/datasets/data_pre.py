import json
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
import torchvision
from PIL import Image
import torch


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_triplets(triplets):
    # Nối tất cả các triplet thành một câu dài để token hóa
    text = " ".join(triplets)
    encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    return encoding['input_ids'], encoding['attention_mask']

class CreateDataset(torchvision.datasets.CocoDetection):
    def __init__(self, mode):
        

        with open('/rel.json', 'r') as f:
            all_rels = json.load(f)

        with open('/categories.json', 'r') as f:
            categories = json.load(f)

        

        if mode == 'train':
            self.rel_annotations = all_rels['train']
            with open('train_trip', 'r') as f:
                self.triplets_data = json.load(f)

        elif mode == 'val':
            self.rel_annotations = all_rels['val']
            with open('val_trip', 'r') as f:
                self.triplets_data = json.load(f)
        else:
            self.rel_annotations = all_rels['test']

        self.rel_categories = all_rels['rel_categories']
        self.categories = categories['categories']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, target = super(CreateDataset, self).__getitem__(idx)


        triplets = item['triplet']
        labels = item['label']

        image_id = self.ids[idx]
        rel_target = self.rel_annotations[str(image_id)]
        triplets_txt = []
        rel_labels = []
        for item in rel_target:
            rel_txt = self.rel_categories[item[2]]
            sub = self.categories[target[item[0]]['category_id'] - 1]['name']
            obj = self.categories[target[item[1]]['category_id'] - 1]['name']
            rel_labels.append(rel_txt)
            triplets_txt.append(sub + ' ' + rel_txt + ' ' + obj)
        
        input_ids, attention_mask = tokenize_triplets(triplets)
        label_ids, label_attention_mask = tokenize_triplets(triplets_txt)

        return input_ids, attention_mask, label_ids, label_attention_mask
    

texts = [
    ["boy wearing shirt", "boy has hair", "shirt on boy", "logo on shirt", "shirt has logo", "boy holding plate", "boy has hand", "hand of boy", "head of boy", "man wearing pant"],
    ["clock on pole", "hand on clock", "clock has face", "window on building", "flower in pot", "sign on pole", "tree on sidewalk", "hand on clock"]
    # Các danh sách triplet khác...
]

# Token hóa các triplet
tokenized_triplets = [tokenize_triplets(triplets) for triplets in texts]

# Chuyển đổi các tokenized triplets thành các tensor đầu vào cho mô hình
input_ids = torch.cat([item[0] for item in tokenized_triplets], dim=0)
attention_mask = torch.cat([item[1] for item in tokenized_triplets], dim=0)

# Đầu vào này sẽ được sử dụng trong TextEncoder
# text_features = text_encoder(input_ids, attention_mask)
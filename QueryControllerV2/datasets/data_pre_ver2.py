import json
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
import torchvision
from PIL import Image
import torch
from pycocotools.coco import COCO


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_triplets(triplets):
    # Nối tất cả các triplet thành một câu dài để token hóa
    text = " ".join(triplets)
    encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    return encoding['input_ids'], encoding['attention_mask']

class CreateDataset(Dataset):
    def __init__(self, mode, transform = None):
        self.im_dir = 'Datasets/VisualGenome/VG_100K/'
        self.transform = transform
        if mode == 'train':
            with open('Datasets/VisualGenome/train_trip.json', 'r') as f:
                self.triplets_data = json.load(f)

        elif mode == 'val':
            with open('Datasets/VisualGenome/val_trip.json', 'r') as f:
                self.triplets_data = json.load(f)

    def __len__(self):
        return len(self.triplets_data)

    def __getitem__(self, idx):
        item = self.triplets_data[idx]

        triplets = item['triplet']
        image_id = item['image_id']

        image = Image.open(self.im_dir + image_id).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        input_ids, attention_mask = tokenize_triplets(triplets)

        return image, input_ids, attention_mask


def build_data(mode):
    data = CreateDataset(mode, transform=preprocess)
    return data


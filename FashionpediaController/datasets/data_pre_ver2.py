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










# texts = [
#     ["boy wearing shirt", "boy has hair", "shirt on boy", "logo on shirt", "shirt has logo", "boy holding plate", "boy has hand", "hand of boy", "head of boy", "man wearing pant"],
#     ["clock on pole", "hand on clock", "clock has face", "window on building", "flower in pot", "sign on pole", "tree on sidewalk", "hand on clock"]
#     # Các danh sách triplet khác...
# ]

# # Token hóa các triplet
# tokenized_triplets = [tokenize_triplets(triplets) for triplets in texts]

# # Chuyển đổi các tokenized triplets thành các tensor đầu vào cho mô hình
# input_ids = torch.cat([item[0] for item in tokenized_triplets], dim=0)
# attention_mask = torch.cat([item[1] for item in tokenized_triplets], dim=0)

# # Đầu vào này sẽ được sử dụng trong TextEncoder
# # text_features = text_encoder(input_ids, attention_mask)
import torch
# from torch_geometric.data import Data
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import json
import ConfigArgs as args
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from RelTR.util.misc import nested_tensor_from_tensor_list
import random

class ConvertGrayToRGB(object):
    def __call__(self, image):
        if image.mode != 'RGB':
            return image.convert('RGB')
        return image

transform = T.Compose([
    T.Resize(512),
    ConvertGrayToRGB(),  # Chuyển ảnh xám thành RGB
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Tạo một instance của BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class LoadData(Dataset):
    def __init__(self, data, image_foler_name, transform):
        self.data = data
        self.image_foler_name = image_foler_name
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_item = self.data[index]

        image = os.path.join(self.image_foler_name, str(data_item['image_id']) + '.jpg')
        image = Image.open(image)

        queries = ', '.join(data_item['queries'])

        queries = transform_sg_to_encoder(queries)
        
        if self.transform:
            image = self.transform(image)

        # Create negative sample
        neg_index = random.randint(0, len(self.data) - 1)
        while neg_index == index:  # Ensure it's not the same as the positive sample
            neg_index = random.randint(0, len(self.data) - 1)
        neg_data_item = self.data[neg_index]
        neg_queries = ', '.join(neg_data_item['queries'])
        neg_queries_encoded = transform_sg_to_encoder(neg_queries)

        return image, queries, neg_queries_encoded

def get_file(file_path):
    file_data = open(file_path)
    data_item = json.load(file_data)
    return data_item

def transform_sg_to_encoder(txt):
    encoded_dict = tokenizer.encode_plus(
                        txt,                  
                        add_special_tokens=True,   # Add '[CLS]' and '[SEP]'
                        max_length=256,             # Adjust sentence length
                        pad_to_max_length=True,    # Pad/truncate sentences
                        return_attention_mask=True,# Generate attention masks
                        return_tensors='pt',       # Return PyTorch tensors
                   )
    
    return encoded_dict['input_ids']

def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    batch[1] = torch.stack(batch[1])
    batch[1] = torch.squeeze(batch[1], dim=1)
    return tuple(batch)

def build_data(mode):
    if mode == 'train':
        all_data = get_file(args.anno_train)
    if mode == 'val':
        all_data = get_file(args.anno_valid)

    dataset = LoadData(all_data, args.img_folder_vg, transform)
    len_data = dataset.__len__()
    print(f'Loaded {len_data} from {mode} ')
    return dataset

dataset_train = build_data('train')
dataset_valid = build_data('val')

dataset_train.__getitem__(100)

sampler_train = torch.utils.data.RandomSampler(dataset_train)
sampler_val = torch.utils.data.SequentialSampler(dataset_valid)
batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers, collate_fn=collate_fn)
data_loader_valid = DataLoader(dataset_valid, args.batch_size, sampler=sampler_val, num_workers=args.num_workers, collate_fn=collate_fn)

for graph_data, text_data_1, text_data_2 in data_loader_train:
    # print(graph_data.size())
    print(text_data_1.size())
    print(text_data_2.size())
    break
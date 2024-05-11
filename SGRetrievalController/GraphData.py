import torch
from torch_geometric.data import Data
from collections import defaultdict
from torch.utils.data import Dataset
import os
import json
import ConfigArgs as args
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

# Tạo một instance của BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class LoadData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_item = self.data[index]
        scene_graph_data, query = transform_sg_to_encoder(data_item)
        return scene_graph_data, query
    


def get_file(file_path):
    files = os.listdir(file_path)
    data = []
    for item in files:
        file_data = open(file_path + item)
        data_item = json.load(file_data)
        data.append(data_item)
    return data

def transform_sg_to_encoder(data_item):
    assert "subject" and "object" and "relation" in data_item.keys(), "Please use correct format. "
    sg_to_str = " "
    for sub, obj, rel in zip(data_item["subject"], data_item["object"], data_item['relation']):
        sg_to_str = sg_to_str + sub + ' ' + rel + ' ' + obj + ', '
    
    scene_graph_data = sg_to_str
    query = data_item['query']
    input_sg = tokenizer(scene_graph_data, return_tensors='pt', padding=True, truncation=True, max_length=args.max_length)
    input_qu = tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=args.max_length)

    return input_sg['input_ids'].squeeze(0), input_qu['input_ids'].squeeze(0)

def collate_fn(batch):
    input_ids1, input_ids2 = zip(*batch)

    padded_input_ids1 = pad_sequence(input_ids1, batch_first=True, padding_value=0)

    padded_input_ids2 = pad_sequence(input_ids2, batch_first=True, padding_value=0)

    return padded_input_ids1, padded_input_ids2

def build_data(mode):
    if mode == 'train':
        all_data = get_file(args.anno_train)
    if mode == 'val':
        all_data = get_file(args.anno_valid)

    dataset = LoadData(all_data)
    len_data = dataset.__len__()
    print(f'Loaded {len_data} from {mode} ')
    return dataset

# dataset_train = build_data('train')
# dataset_valid = build_data('val')

# print(dataset_train.__getitem__(100))
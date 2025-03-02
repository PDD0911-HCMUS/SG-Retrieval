import json
from torch.utils.data import Dataset, random_split
import torch
from transformers import BertTokenizer
from torch import Tensor
from typing import List

from torch import Tensor

class CreateData(Dataset):
    def __init__(self, ann_file):

        with open(ann_file, 'r') as f:
            self.data = json.load(f)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.maxlenght = 10

    def __getitem__(self, idx):
        encoded_triplets_que = []
        encoded_triplets_rev = []

        labels_que_s = []
        labels_que_o = []
        labels_que_r = []

        labels_rev_s = []
        labels_rev_o = []
        labels_rev_r = []

        triplets_que = {}
        triplets_rev = {}

        que = self.data[idx]['qe']
        rev = self.data[idx]['rev']

        for item_que in que['trip']:
            words = item_que.split(' ')
            encoded_triplets_que.append(item_que)
            labels_que_s.append(words[0])
            labels_que_o.append(words[-1])
            labels_que_r.append(' '.join(words[1:-1]))

        for item_rev in rev['trip']:
            words = item_rev.split(' ')
            encoded_triplets_rev.append(item_rev)
            labels_rev_s.append(words[0])
            labels_rev_o.append(words[-1])
            labels_rev_r.append(' '.join(words[1:-1]))

        labels_que_s = [' '.join(labels_que_s)]
        labels_que_o = [' '.join(labels_que_o)]
        labels_que_r = [' '.join(labels_que_r)]

        
        
        labels_rev_s = [' '.join(labels_rev_s)]
        labels_rev_o = [' '.join(labels_rev_o)]
        labels_rev_r = [' '.join(labels_rev_r)]

        e_s_que = self.tokenizer(labels_que_s, padding = 'max_length', truncation = True, max_length = self.maxlenght, return_tensors = 'pt')
        e_o_que = self.tokenizer(labels_que_o, padding = 'max_length', truncation = True, max_length = self.maxlenght, return_tensors = 'pt')
        e_r_que = self.tokenizer(labels_que_r, padding = 'max_length', truncation = True, max_length = self.maxlenght, return_tensors = 'pt')
        e_t_que = self.tokenizer(encoded_triplets_que, padding = 'max_length', truncation = True, max_length = self.maxlenght, return_tensors = 'pt')

        e_s_rev = self.tokenizer(labels_rev_s, padding = 'max_length', truncation = True, max_length = self.maxlenght, return_tensors = 'pt')
        e_o_rev = self.tokenizer(labels_rev_o, padding = 'max_length', truncation = True, max_length = self.maxlenght, return_tensors = 'pt')
        e_r_rev = self.tokenizer(labels_rev_r, padding = 'max_length', truncation = True, max_length = self.maxlenght, return_tensors = 'pt')
        e_t_rev = self.tokenizer(encoded_triplets_rev, padding = 'max_length', truncation = True, max_length = self.maxlenght, return_tensors = 'pt')

        triplets_que['sub_labels'] = e_s_que['input_ids']
        triplets_que['obj_labels'] = e_o_que['input_ids']
        triplets_que['rel_labels'] = e_r_que['input_ids']
        triplets_que['trip'] = e_t_que['input_ids']
        triplets_que['sub_labels_msk'] = e_s_que['attention_mask']
        triplets_que['obj_labels_msk'] = e_o_que['attention_mask']
        triplets_que['rel_labels_msk'] = e_r_que['attention_mask']
        triplets_que['trip_msk'] = e_t_que['attention_mask']


        triplets_rev['sub_labels'] = e_s_rev['input_ids']
        triplets_rev['obj_labels'] = e_o_rev['input_ids']
        triplets_rev['rel_labels'] = e_r_rev['input_ids']
        triplets_rev['trip'] = e_t_rev['input_ids']
        triplets_rev['sub_labels_msk'] = e_s_rev['attention_mask']
        triplets_rev['obj_labels_msk'] = e_o_rev['attention_mask']
        triplets_rev['rel_labels_msk'] = e_r_rev['attention_mask']
        triplets_rev['trip_msk'] = e_t_rev['attention_mask']

        return triplets_que, triplets_rev
    
    def __len__(self):
        return len(self.data) 

def pad_or_truncate_tensor(item, max_i = 10):
    # Lấy kích thước hiện tại của tensor
    for i in item.keys():
        if(i == 'trip' or i == 'trip_msk'):
            if(item[i].size(0) < max_i):
                num_i, _ = item[i].size()
                expand = torch.zeros((max_i - num_i, max_i), dtype=item[i].dtype)
                item[i] = torch.cat([item[i], expand], dim = 0)
            if(item[i].size(0) > max_i):
                item[i] = item[i][: max_i]
    return item

def process_batch(tensor_list: List[Tensor]):
    for item in tensor_list:
        item = pad_or_truncate_tensor(item)
    return tensor_list

def custom_collate_fn(batch):
    batch = list(zip(*batch))

    batch[0] = process_batch(batch[0])
    batch[1] = process_batch(batch[1])

    return tuple(batch)

def build(ann_file, ratio = 0.8):
    
    dataset = CreateData(ann_file=ann_file)
    train_size = int(ratio * dataset.__len__())
    valid_size = dataset.__len__() - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    return train_dataset, valid_dataset
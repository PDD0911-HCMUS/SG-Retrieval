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

        self.maxlenght = 10

    def __getitem__(self, idx):
        encoded_triplets_que = []
        encoded_triplets_rev = []

        triplets_que = {}
        triplets_rev = {}

        que = self.data[idx]['qe']
        rev = self.data[idx]['rev']

        que_im = self.data[idx]['qe']["image_id"]
        rev_im = self.data[idx]['rev']["image_id"]

        encoded_triplets_que = [", ".join(que['trip'])]
        encoded_triplets_rev = [", ".join(rev['trip'])]

        triplets_que['image_id'] = que_im
        triplets_que['trip'] = encoded_triplets_que

        triplets_rev['image_id'] = rev_im
        triplets_rev['trip'] = encoded_triplets_rev

        return triplets_que, triplets_rev
    
    def __len__(self):
        return len(self.data) 

def build_v2(ann_file):
    
    dataset = CreateData(ann_file=ann_file)
    return dataset
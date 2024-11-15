import json
from torch.utils.data import Dataset, random_split, DataLoader
import torch
from transformers import BertTokenizer

class CreateData(Dataset):
    def __init__(self, ann_file):

        with open(ann_file, 'r') as f:
            self.data = json.load(f)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __getitem__(self, idx):
        encoded_triplets_que = []
        encoded_triplets_rev = []

        labels_que_s = []
        labels_que_o = []
        labels_rev_s = []
        labels_rev_o = []

        triplets_que = {}
        triplets_rev = {}

        que = self.data[idx]['qe']
        rev = self.data[idx]['rev']

        for item_que in que['trip']:
            words = item_que.split(' ')
            encoded_triplets_que.append(item_que)
            labels_que_s.append(words[0])
            labels_que_o.append(words[-1])

        for item_rev in rev['trip']:
            words = item_que.split(' ')
            encoded_triplets_rev.append(item_rev)
            labels_rev_s.append(words[0])
            labels_rev_o.append(words[-1])

        triplets_que['sub_labels_qu'] = self.tokenizer(labels_que_s, padding = 'max_length', truncation = True, max_length = 10, return_tensors = 'pt')['input_ids']
        triplets_que['obj_labels_qu'] = self.tokenizer(labels_que_o, padding = 'max_length', truncation = True, max_length = 10, return_tensors = 'pt')['input_ids']
        # triplets_que['rel_labels'] = encoded_triplets_que[:,1]
        triplets_que['trip_qu'] = self.tokenizer(encoded_triplets_que, padding = 'max_length', truncation = True, max_length = 10, return_tensors = 'pt')['input_ids']

        triplets_rev['sub_labels_rev'] = labels_rev_s
        triplets_rev['obj_labels_rev'] = labels_rev_o
        # triplets_rev['rel_labels'] = encoded_triplets_rev[:,1]
        triplets_rev['trip_rev'] = encoded_triplets_rev


        return triplets_que, triplets_rev
    
    
    def __len__(self):
        return len(self.data) 

def pad_or_truncate_tensor(tensor, target_num_cls=10, target_length=10):
    # Lấy kích thước hiện tại của tensor
    current_num_cls = tensor.size(0)
    
    if current_num_cls < target_num_cls:
        padding = torch.zeros(target_num_cls - current_num_cls, target_length, dtype=tensor.dtype)
        tensor = torch.cat([tensor, padding], dim=0)
    
    return tensor

def custom_collate_fn(batch):
    # Tách các phần tử của batch và đảm bảo mỗi tensor có kích thước [10, 10]
    batch = list(zip(*batch))
    for item in batch[0]:
        print(item)

    return tuple(batch)

def build(ann_file):
    ratio = 0.8
    dataset = CreateData(ann_file=ann_file)
    train_size = int(ratio * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    return train_dataset, valid_dataset

if __name__ == "__main__":
    ann_file = '/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/VisualGenome/Rev.json'
    train_dataset, valid_dataset = build(ann_file)
    idx = 1
    triplets_que, triplets_rev = train_dataset.__getitem__(idx)
    for item in triplets_que.keys():
        print(f'{item}: {triplets_que[item].size()}')

    dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=custom_collate_fn)

    for batch in dataloader:
        print("Batch shape:", batch.shape)
        print(batch)


    # print(f'encoded_triplets_que: {triplets_que}')
    # print(f'encoded_triplets_rev: {triplets_rev}')
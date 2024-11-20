import json
from torch.utils.data import Dataset, random_split, DataLoader
import torch
from transformers import BertTokenizer, BertModel
from torch import Tensor
from typing import Optional, List

from torch import nn, Tensor

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

    return tuple(batch)

def build(ann_file):
    ratio = 0.8
    dataset = CreateData(ann_file=ann_file)
    train_size = int(ratio * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    return train_dataset, valid_dataset

def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "glu":
        return nn.GLU()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

if __name__ == "__main__":
    activation = _get_activation_fn('relu')
    entity_embed = nn.Sequential(
            nn.Linear(768, 512),
            activation,
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256))
    model = BertModel.from_pretrained('bert-base-uncased')
    device = 'cpu'
    ann_file = '/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/VisualGenome/Rev.json'
    train_dataset, valid_dataset = build(ann_file)
    # idx = 1
    # triplets_que, triplets_rev = train_dataset.__getitem__(idx)
    # for item in triplets_que.keys():
    #     print(f'{item}: {triplets_que[item].size()}')

    dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=custom_collate_fn)

    # ck = torch.tensor([[  101, 29294, 22747, 21759, 102], 
    #                    [  101, 29294, 22747, 21759, 102],
    #                    [  101, 29294, 22747, 21759, 102]])
    # ck = torch.stack([ck, ck])
    # print(ck.size())
    # print(ck)
    # ws = []
    # for i in ck:
    #     with torch.no_grad():
    #         outputs = model(i)
    #         word_embeddings = outputs.last_hidden_state 
    #         sub_embeddings = word_embeddings.mean(dim=1)
    #         print(sub_embeddings.size())
    #         ws.append(sub_embeddings)
    # o = torch.stack([g for g in ws])
    # print(o.size())




    for que, rev in dataloader:
        # print(que)
        que = [{k: v.to(device) for k, v in t.items()} for t in que]
        print(len(que))
        # for i in que:
        #     print(i['sub_labels'].size())
        #     print(i['sub_labels'])
        nodes_s = torch.stack([g["trip"] for g in que])
        nodes_s_msk = torch.stack([g["trip_msk"] for g in que])
        # print(nodes_s.size())
        # print(nodes_s)
        # print(nodes_s_msk.size())
        # print(nodes_s_msk)
        word_means = []
        for i,m in zip(nodes_s, nodes_s_msk):
            with torch.no_grad():
                outputs = model(i, attention_mask = m)
                word_em = outputs.last_hidden_state
                word_mean = word_em.mean(dim = 1)
                word_means.append(word_mean)
        z = torch.stack([w for w in word_means])
        z = entity_embed(z)
        print(z.size())
        break

    

    # print(f'encoded_triplets_que: {triplets_que}')
    # print(f'encoded_triplets_rev: {triplets_rev}')



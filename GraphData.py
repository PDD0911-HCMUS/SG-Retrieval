import torch
from torch_geometric.data import Data
from collections import defaultdict
from torch.utils.data import Dataset
import os
import json
import ConfigArgs as args

batch_size = 128
data_root = 'Datasets/Incidents/anno/'

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
    node_to_index = defaultdict(lambda: len(node_to_index))  # Tự động gán chỉ số mới cho mỗi node mới
    # Tạo edge index
    edge_index = []
    sg_to_str = " "
    for sub, obj, rel in zip(data_item["subject"], data_item["object"], data_item['relation']):
        edge_index.append([node_to_index[sub], node_to_index[obj]])
        sg_to_str = sg_to_str + sub + ' ' + rel + ' ' + obj + ','
    num_nodes = len(node_to_index)
    node_features = torch.eye(num_nodes)
    scene_graph_data = Data(x=node_features, edge_index=edge_index)
    query = [data_item['query']]
    return scene_graph_data, query

def build_data(mode):
    if mode == 'train':
        all_data = get_file(args.anno_train)
    if mode == 'val':
        all_data = get_file(args.anno_valid)

    dataset = LoadData(all_data)
    len_data = dataset.__len__()
    print(f'Loaded {len_data} from {mode} ')
    return dataset

dataset_train = build_data('train')
dataset_valid = build_data('val')
import json
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
import torchvision
from PIL import Image
import torch
from pycocotools.coco import COCO
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def is_exist_im(folder_im, file_name):
    classes = ['animals', 'collapse', 'crash', 'fire', 'flooding', 'landslide', 'snow', 'treefall']
    cls_return = None
    cls_qu = None
    for cls in classes:
        path = os.path.join(folder_im, cls, file_name)
        if(os.path.exists(path)):
            cls_return = cls
            cls_qu = f'this is an incident of {cls_return}'
            return cls_return, cls_qu, file_name
        else:
            return False

def link_triplets(subs, objs, rels):
    linked = []
    for s, o, r in zip(subs, objs, rels):
        linked.append(s + " " + o + " " + r)
    return linked

def create_anno(json_files, folder_im, mode):
    classes = ['animals', 'collapse', 'crash', 'fire', 'flooding', 'landslide', 'snow', 'treefall']
    cls_return = None
    cls_qu = None

    data_anno = []
    for json_file in tqdm(json_files):
        with open(os.path.join('Datasets/Incidents/', mode, json_file), 'r') as f:
            json_item = json.load(f)
            for cls in classes:
                path = os.path.join(folder_im, cls, json_item['file_name'])
                if(os.path.exists(path)):
                    cls_return = cls
                    cls_qu = f'this is an incident of {cls_return}'
                    triplets = link_triplets(json_item['subject'], json_item['object'], json_item['relation'])
                    data_item = {
                        "file_name": json_item['file_name'],
                        "annotation":{
                            "triplets": triplets,
                            "cls_query": cls_qu,
                            "query": json_item['query']
                        }
                    }
                    data_anno.append(data_item)
                    break

    with open(f'incidents_{mode}.json', 'w') as f:
        json.dump(data_anno, f)
    
def tokenize_triplets(triplets):
    
    text = " ".join(triplets)
    encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    return encoding['input_ids'], encoding['attention_mask']

class CreateDataset(Dataset):
    def __init__(self, mode, annotation_file):
        

        with open('Datasets/VisualGenome/rel.json', 'r') as f:
            all_rels = json.load(f)

        with open('Datasets/VisualGenome/categories.json', 'r') as f:
            categories = json.load(f)

        self.coco = COCO(annotation_file)

        if mode == 'train':
            self.rel_annotations = all_rels['train']
            with open('Datasets/VisualGenome/train_trip.json', 'r') as f:
                self.triplets_data = json.load(f)

        elif mode == 'val':
            self.rel_annotations = all_rels['val']
            with open('Datasets/VisualGenome/val_trip.json', 'r') as f:
                self.triplets_data = json.load(f)
        else:
            self.rel_annotations = all_rels['test']

        self.rel_categories = all_rels['rel_categories']
        self.categories = categories['categories']

    def __len__(self):
        return len(self.triplets_data)

    def __getitem__(self, idx):
        item = self.triplets_data[idx]

        triplets = item['triplet']
        image_id = item['image_id']
        rel_target = self.rel_annotations[str(image_id).replace('.jpg', '')]

        annotation_ids = self.coco.getAnnIds(imgIds=[int(image_id.replace('.jpg', ''))])
        anno_coco = self.coco.loadAnns(annotation_ids)

        triplets_txt = []
        rel_labels = []
        for item in rel_target:
            rel_txt = self.rel_categories[item[2]]
            sub = self.categories[anno_coco[item[0]]['category_id'] - 1]['name']
            obj = self.categories[anno_coco[item[1]]['category_id'] - 1]['name']

            rel_labels.append(rel_txt)
            triplets_txt.append(sub + ' ' + rel_txt + ' ' + obj)

        # print(f'triplets promt: {triplets}')
        # print(f'triplets: {triplets_txt}')
        
        input_ids, attention_mask = tokenize_triplets(triplets)
        label_ids, label_attention_mask = tokenize_triplets(triplets_txt)

        return input_ids, attention_mask, label_ids, label_attention_mask


def build_data(mode):
    if(mode == 'train'):
        annotation_file = 'Datasets/VisualGenome/train.json'
    if(mode == 'val'):
        annotation_file = 'Datasets/VisualGenome/val.json'
    
    data = CreateDataset(mode, annotation_file)
    return data

if __name__ == "__main__":
    data_dir = 'Datasets/Incidents/'
    folder_im = 'Datasets/Incidents/incidents_by_class/'
    mode = 'val'

    if(mode == 'train'):
        data_dir = data_dir + 'train/'
        json_files = os.listdir(data_dir)
        create_anno(json_files, folder_im, mode)
    if(mode == 'val'):
        data_dir = data_dir + 'val/'
        json_files = os.listdir(data_dir)
        create_anno(json_files, folder_im, mode)
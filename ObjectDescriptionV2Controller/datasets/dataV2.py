import json
from torch.utils.data import Dataset, random_split
import torch

class CreateData(Dataset):
    def __init__(self, ann_file):

        with open(ann_file, 'r') as f:
            self.data = json.load(f)

        rel_categories = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']


        classes = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
                'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
                'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
                'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
                'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
                'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
                'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
                'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
                'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
                'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
                'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
                'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
                'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
                'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']
        
        self.class_to_idx = {obj: idx for idx, obj in enumerate(classes)}
        self.rel_to_idx = {rel: idx for idx, rel in enumerate(rel_categories)}

    

    def __getitem__(self, idx):
        encoded_triplets_que = []
        encoded_triplets_rev = []

        que = self.data[idx]['qe']
        rev = self.data[idx]['rev']

        for item_que in que['trip']:
            obj1, rel, obj2 = item_que.split(' ')[0], item_que.split(' ')[1], item_que.split(' ')[2]
            obj1_idx = self.class_to_idx.get(obj1, 0)
            rel_idx = self.rel_to_idx.get(rel, 0)
            obj2_idx = self.class_to_idx.get(obj2, 0)

            encoded_triplets_que.append([obj1_idx, rel_idx, obj2_idx])

        for item_rev in rev['trip']:
            obj1, rel, obj2 = item_rev.split(' ')[0], item_rev.split(' ')[1], item_rev.split(' ')[2]
            obj1_idx = self.class_to_idx.get(obj1, 0)
            rel_idx = self.rel_to_idx.get(rel, 0)
            obj2_idx = self.class_to_idx.get(obj2, 0)

            encoded_triplets_rev.append([obj1_idx, rel_idx, obj2_idx])

        while len(encoded_triplets_que) < 7:
            encoded_triplets_que.append([0, 0, 0])

        while len(encoded_triplets_rev) < 7:
            encoded_triplets_rev.append([0, 0, 0])
        
        encoded_triplets_que = torch.tensor(encoded_triplets_que)
        encoded_triplets_rev = torch.tensor(encoded_triplets_rev)

        return encoded_triplets_que, encoded_triplets_rev
    
    def __len__(self):
        return len(self.data) 


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
    encoded_triplets_que, encoded_triplets_rev = train_dataset.__getitem__(idx)
    print(f'encoded_triplets_que: {encoded_triplets_que}')
    print(f'encoded_triplets_rev: {encoded_triplets_rev}')
import json
import torch
from torch.utils.data import Dataset, random_split
from transformers import BertTokenizer
from typing import List, Tuple, Dict
from PIL import Image
import os
import Controller.HybridEncoderRegionDescriptionController.datasets.transforms as T

class HybridEncodeData(Dataset):

    def __init__(self, anno_file, image_folder, transforms, tokenizer):
        with open(anno_file, 'r') as f:
            self.data = json.load(f)
        self.image_folder = image_folder
        self.transforms = transforms
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)

    def __getitem__(self, idx: int):
        img, targets = self.get_info(self.data[idx])
        if self.transforms is not None:
            img, targets = self.transforms(img, targets)
        return img, targets
    
    def __len__(self):
        return len(self.data)

    def get_info(self, data):
        im_id = str(data['image_id']) + '.jpg'
        img = Image.open(os.path.join(self.image_folder, im_id)).convert('RGB')
        w, h = img.size
        anno = data['regions']
        regions = [re['phrase'] for re in anno]

        boxes = [b['bbox'] for b in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        targets = {
            'image_id': im_id,
            'boxes': boxes,
            'regions': regions,
            'orig_size': torch.as_tensor([int(h), int(w)]),
            'size': torch.as_tensor([int(h), int(w)])
        }

        return img, targets
    
def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640]

    if image_set == 'train':
        return T.Compose([
            #T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=640),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    #T.RandomSizeCrop(384, 500),
                    T.RandomResize(scales, max_size=640),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([512], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build_data(image_folder, anno_file, tokenizer, image_set):
    dataset = HybridEncodeData(
        anno_file=anno_file,
        image_folder=image_folder,
        tokenizer=tokenizer,
        transforms=make_coco_transforms(image_set)
    )
    return dataset
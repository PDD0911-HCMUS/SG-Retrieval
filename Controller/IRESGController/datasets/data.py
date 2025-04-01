import json
import torch
from torch.utils.data import Dataset, random_split
from transformers import BertTokenizer
from typing import List, Tuple, Dict
from PIL import Image
import os
from util.misc import nested_tensor_from_tensor_list
import Controller.IRESGController.datasets.transform as T

class CreateData(Dataset):
    def __init__(self, image_folder, transforms, ann_file: str, tokenizer: str, max_length: int = 10):
        with open(ann_file, 'r') as f:
            self.data = json.load(f)

        self.img_folder = image_folder
        self._transforms = transforms
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length

    def encode_triplets(self, triplets: List[str]) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            triplets,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'trip': enc['input_ids'],         # shape: [num_triplets, max_len]
            'trip_msk': enc['attention_mask'] # shape: [num_triplets, max_len]
        }

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        image_id_a = self.data[idx]['qe']['image_id']
        image_id_b = self.data[idx]['rev']['image_id']

        img_a = Image.open(os.path.join(self.img_folder, image_id_a)).convert('RGB')
        img_b = Image.open(os.path.join(self.img_folder, image_id_b)).convert('RGB')

        if self._transforms is not None:
            img_a, _ = self._transforms(img_a, target = None)
            img_b, _ = self._transforms(img_b, target = None)

        triplets_que = self.encode_triplets(self.data[idx]['qe']['trip'])
        triplets_rev = self.encode_triplets(self.data[idx]['rev']['trip'])
        return img_a, img_b, triplets_que, triplets_rev

    def __len__(self) -> int:
        return len(self.data)
    
def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544]

    if image_set == 'train':
        return T.Compose([
            #T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=544),
                T.Compose([
                    T.RandomResize([400, 500]),
                    #T.RandomSizeCrop(384, 500),
                    T.RandomResize(scales, max_size=544),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([512], max_size=544),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def pad_or_truncate_tensor(item: Dict[str, torch.Tensor], max_i: int = 10) -> Dict[str, torch.Tensor]:
    for key in ['trip', 'trip_msk']:
        if key in item:
            seq = item[key]
            if seq.size(0) < max_i:
                padding = torch.zeros((max_i - seq.size(0), seq.size(1)), dtype=seq.dtype)
                item[key] = torch.cat([seq, padding], dim=0)
            elif seq.size(0) > max_i:
                item[key] = seq[:max_i]
    return item

def process_batch(tensor_list: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    return [pad_or_truncate_tensor(item) for item in tensor_list]

def collate_fn_dual_image(batch):
    # batch = [(I_a, I_b, triplets_que, triplets_rev), ...]
    images_a, images_b, triplets_que_list, triplets_rev_list = zip(*batch)

    # NestedTensor từ DETR
    images_a = nested_tensor_from_tensor_list(images_a)
    images_b = nested_tensor_from_tensor_list(images_b)

    # Xử lý token triplets
    triplets_que_list = process_batch(list(triplets_que_list))
    triplets_rev_list = process_batch(list(triplets_rev_list))

    return images_a, images_b, triplets_que_list, triplets_rev_list

def build_data(image_folder, ann_file, tokenizer, max_length, image_set):

    dataset = CreateData(image_folder=image_folder,
                        transforms=make_coco_transforms(image_set),
                        ann_file=ann_file,
                        tokenizer=tokenizer,
                        max_length=max_length
                        )
    return dataset
    

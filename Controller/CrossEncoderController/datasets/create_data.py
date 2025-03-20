import Controller.CrossEncoderController.datasets.transform as T
import json
from torch.utils.data import Dataset
import torch
import gc
import os
from PIL import Image
from transformers import AutoTokenizer

class CreateData(Dataset):
    def __init__(self, img_folder, ann_file, transforms, tokenizer):
        with open(ann_file, 'r') as f:
            self.data = json.load(f)

        self.img_folder = img_folder
        self._transforms = transforms
        self.prepare = PrepareImageTarget(tokenizer)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        img_id = self.data[idx]['image_id']
        img_path = os.path.join(self.img_folder,str(img_id) + '.jpg')

        img = Image.open(img_path).convert('RGB')
        target = self.data[idx]['regions']

        img, target = self.prepare(img, target)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        img_id = torch.tensor([img_id])
        target['image_id'] = img_id
        return img, target
    
class PrepareImageTarget(object):
    def __init__(self, tokenizer):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_boxes = 10
        
    def __call__(self, img, target):
        w,h = img.size

        bbox_rg =  [obj["bbox"] for obj in target]
        phrases = [obj["phrase"] for obj in target]
        bbox_rg = torch.as_tensor(bbox_rg, dtype=torch.float32).reshape(-1, 4)

        # Convert bounding box format from (x, y, width, height) to (x_min, y_min, x_max, y_max)
        # x_max = x_min + width
        # y_max = y_min + height
        bbox_rg[:, 2:] += bbox_rg[:, :2]

        # Clamp bounding box coordinates to ensure they stay within the image dimensions
        # boxes[:, 0::2] selects x_min and x_max, ensuring they are within [0, w]
        # boxes[:, 1::2] selects y_min and y_max, ensuring they are within [0, h]
        bbox_rg[:, 0::2].clamp_(min=0, max=w)
        bbox_rg[:, 1::2].clamp_(min=0, max=h)

        # Filter valid bboxes (width > 0, height > 0)
        keep = (bbox_rg[:, 3] > bbox_rg[:, 1]) & (bbox_rg[:, 2] > bbox_rg[:, 0])
        bbox_rg = bbox_rg[keep]
        phrases = [phrases[i] for i in keep.nonzero(as_tuple=True)[0].tolist()]

        # Arrange bounding boxes by area from largest to smallest
        areas = (bbox_rg[:, 2] - bbox_rg[:, 0]) * (bbox_rg[:, 3] - bbox_rg[:, 1])
        sorted_indices = torch.argsort(areas, descending=True)
        bbox_rg = bbox_rg[sorted_indices]
        phrases = [phrases[i] for i in sorted_indices.tolist()]

        bbox_rg = bbox_rg[:self.max_boxes]
        phrases = phrases[:self.max_boxes]

        phrase_tok = self.tokenizer(
            phrases, padding="max_length", truncation=True, max_length=10, return_tensors="pt", add_special_tokens=True
        )
        phrase_ids = phrase_tok['input_ids']
        phrase_msk = phrase_tok['attention_mask']

        num_boxes = bbox_rg.shape[0]
        if num_boxes < self.max_boxes:
            pad_size = self.max_boxes - num_boxes
            
            # Padding boxes with 0
            pad_boxes = torch.zeros((pad_size, 4), dtype=torch.float32)
            bbox_rg = torch.cat([bbox_rg, pad_boxes], dim=0)

            # Padding phrases with [PAD]
            pad_ids = torch.full((pad_size, phrase_ids.shape[1]), self.tokenizer.pad_token_id, dtype=torch.long)
            pad_mask = torch.zeros((pad_size, phrase_msk.shape[1]), dtype=torch.long)
            phrase_ids = torch.cat([phrase_ids, pad_ids], dim=0)
            phrase_msk = torch.cat([phrase_msk, pad_mask], dim=0)

        target = {}
        target['image_id'] = bbox_rg
        target['boxes'] = bbox_rg
        target['phrase_ids'] = phrase_ids
        target['phrase_msk'] = phrase_msk
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return img, target

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

def build_data(image_set, annotation_file, image_folder, tokenizer):
    dataset = CreateData(image_folder, annotation_file, transforms=make_coco_transforms(image_set), tokenizer = tokenizer)
    return  dataset




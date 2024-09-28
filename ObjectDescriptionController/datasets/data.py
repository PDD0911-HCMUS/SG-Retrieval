# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from transformers import BertTokenizer
import torch
import torch.utils.data
import torchvision

import datasets.transforms as T
import util.config as cf

from PIL import Image, ImageDraw 
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import re
import random

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCoco(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

class ConvertCoco(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size
        

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        desc = [re.sub(r'[^A-Za-z0-9\s]', '', obj["caption"]) for obj in anno]

        token = tokenizer(desc, padding='max_length', truncation=True, max_length=7, return_tensors="pt")

        desc_emb = token['input_ids']
        desc_msk = token['attention_mask']

        target = {}

        target["image_id"] = image_id

        target["desc"] = desc
        target["desc_emb"] = desc_emb
        target["desc_msk"] = desc_msk

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [128, 256]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=256),
                T.Compose([
                    T.RandomResize([64, 128, 256]),
                    # T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=256),
                ])
            ),
            normalize,
        ])
    if image_set == 'val':
        return T.Compose([
            T.RandomResize([128], max_size=256),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_dataset(image_set):
    if (image_set == 'train'):
        ann_file = cf.data_anno_train
        img_folder = cf.data_image
    if (image_set == 'val'):
        ann_file = cf.data_anno_valid
        img_folder = cf.data_image
    
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=False)
    return dataset

def imshow(img):
    img = img.numpy().transpose((1, 2, 0))  # chuyển từ tensor sang numpy
    img = np.clip(img, 0,1)
    return img

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    print(size)
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def build_example():
    trainDataset = build_dataset('train')
    print(len(trainDataset))
    # index = random.randint(0, 1000)
    index = 627
    imageTransform, target = trainDataset.__getitem__(index)
    print(imageTransform.shape)
    print(target)
    imCopy = imshow(imageTransform)
    imgPil = Image.fromarray((imCopy * 255).astype(np.uint8))
    imageDraw = ImageDraw.Draw(imgPil)
    print(tokenizer.vocab_size)
    plt.figure(figsize=(16,10))
    plt.imshow(imgPil)
    plt.show()


# if __name__ == "__main__":
#     build_example()
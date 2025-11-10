# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
import os
import datasets.transforms as T
# from config_run import *

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, image_set, root_image_seg_folder, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask()

        self.image_seg = os.path.join(root_image_seg_folder, image_set)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        image_name = self.coco.loadImgs(image_id)[0]['file_name']
        seg_path = os.path.join(self.image_seg, image_name.replace('.jpg', '.png'))
        target_seg = Image.open(seg_path)
        target_seg = np.array(target_seg) 
        target_seg = torch.as_tensor(target_seg, dtype=torch.uint8).unsqueeze(0)
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target, target_seg)
        target['masks'] = target_seg
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

class ConvertCocoPolysToMask(object):
    def __init__(self):
        pass
    def __call__(self, image, target, target_seg):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        # target['orig_seg_size'] = torch.as_tensor([int(h_seg), int(w_seg)])
        # target['seg_size'] = torch.as_tensor([int(h_seg), int(w_seg)])

        return image, target


def make_coco_transforms(image_set, size):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [size,size]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=size),
                T.Compose([
                    T.RandomResize([size,size]),
                    # T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=size),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([size,size], max_size=size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, root_image_folfer, root_anno_folder,  root_image_seg_folder, size):
    mode = 'instances'
    if(image_set == "train"):
        img_folder = os.path.join(root_image_folfer, image_set)
        ann_file = os.path.join(root_anno_folder, f"{mode}_{image_set}2017.json")
    if(image_set == "val"):
        img_folder = os.path.join(root_image_folfer, image_set)
        ann_file = os.path.join(root_anno_folder, f"{mode}_{image_set}2017.json")
    dataset = CocoDetection(img_folder, ann_file, image_set, root_image_seg_folder, transforms=make_coco_transforms(image_set, size))
    return dataset
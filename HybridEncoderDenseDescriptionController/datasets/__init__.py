# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


# def build_dataset(image_set, args):
#     if args.dataset == 'vg' or args.dataset == 'oi':
#         return build_coco(image_set, args)
#     raise ValueError(f'dataset {args.dataset} not supported')


def build_dataset(image_set):
    
    return build_coco(image_set)
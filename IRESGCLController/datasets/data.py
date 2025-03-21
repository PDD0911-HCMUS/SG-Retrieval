# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

"""
dataset (COCO-like) which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import json
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
import numpy as np
import datasets.transforms as T
import util.misc as utils
from torch.utils.data import Dataset, DataLoader, DistributedSampler
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

        #TODO load relationship
        with open('/'.join(ann_file.split('/')[:-1])+'/rel.json', 'r') as f:
            all_rels = json.load(f)
        if 'train' in ann_file:
            self.rel_annotations = all_rels['train']
        elif 'val' in ann_file:
            self.rel_annotations = all_rels['val']
        else:
            self.rel_annotations = all_rels['test']

        self.rel_categories = all_rels['rel_categories']

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        rel_target = np.array(self.rel_annotations[str(image_id)])

        rel_target = rel_target[np.unique(rel_target[:,:2], return_index=True, axis=0)[1]] # remove duplicates
        np.random.shuffle(rel_target)
        sampled_entities = []
        sampled_triplets= []
        i = 0
        while len(sampled_entities)<10 and len(sampled_triplets)<len(rel_target):
            b1, b2, _ = rel_target[i]
            if len(np.unique(sampled_entities+[b1,b2])) <= 10:
                if b1 not in sampled_entities:
                    sampled_entities.append(b1)
                if b2 not in sampled_entities:
                    sampled_entities.append(b2)
                sampled_triplets.append(rel_target[i])
                i += 1
            else:
                break

        np.random.shuffle(sampled_entities)
        sampled_entities = list(sampled_entities)
        reindex_triplets = []
        for triplet in sampled_triplets:
            reindex_triplets.append([sampled_entities.index(triplet[0]),
                                     sampled_entities.index(triplet[1]),
                                     triplet[2]])
        sampled_triplets = reindex_triplets

        target = {'image_id': image_id, 'annotations': target, 'rel_annotations': sampled_triplets}

        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # reorder boxes, labels
        target['boxes'] = target['boxes'][sampled_entities]
        target['labels'] = target['labels'][sampled_entities]

        if target['boxes'].shape[0] < 10:
            target['boxes'] = torch.cat([target['boxes'], torch.zeros([10-target['boxes'].shape[0], 4])], dim=0) # padding to 10
            target['labels'] = torch.cat([target['labels'], 151*torch.ones(10-target['labels'].shape[0], dtype=torch.int64)], dim=0)# padding to 10, class index 151
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
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

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        # TODO add relation gt in the target
        rel_annotations = target['rel_annotations']

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        # TODO add relation gt in the target
        target['rel_annotations'] = torch.tensor(rel_annotations)

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.Resize(size=512),
            normalize])

    if image_set == 'val':
        return T.Compose([
            T.Resize(size=512),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set):

    ann_path = '/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/VisualGenome/anno_reltr/'
    img_folder = '/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/VisualGenome/VG_100K/'
    #TODO: adapt vg as coco
    if image_set == 'train':
        ann_file = ann_path + 'train.json'
    elif image_set == 'val':
        ann_file = ann_path + 'val.json'
        # if args.eval:
        #     ann_file = ann_path + 'test.json'
        # else:
        #     ann_file = ann_path + 'val.json'

    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=False)
    return dataset

def pre_processing(targets):
    edge_max_num = max([len(t['rel_annotations']) for t in targets])
    for t in targets:
        if t['rel_annotations'].shape[0] < edge_max_num:
            t['rel_annotations'] = torch.cat([t['rel_annotations'],
                                              torch.tensor([[0, 0, 51]],
                                               dtype=torch.long,
                                               device=t['rel_annotations'].device).repeat(
                                                edge_max_num - t['rel_annotations'].shape[0], 1)], dim=0)

    return targets

if __name__ == "__main__":
    device = 'cpu'
    dataset = build("val")
    idx = 1
    img, target = dataset.__getitem__(idx)
    data_loader_val = DataLoader(dataset, 2, drop_last=True, collate_fn=utils.collate_fn)
    # print(target)
    for samples, targets in data_loader_val:
        # print(targets)
        print(type(targets))
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        print(targets)
        targets = pre_processing(targets)
        print(200*'=')
        print(targets)
        break
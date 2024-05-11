# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

"""
dataset (COCO-like) which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import os
import sys
get_pwd = os.getcwd()
sys.path.insert(0, get_pwd)
from pathlib import Path
import torch
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt
import Datasets.TransformUtils as T
import numpy as np
import ConfigArgs as args
from PIL import Image, ImageDraw 
from matplotlib import cm
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        print(f'Getting item {idx}, image_id {image_id}')
        print(f'Original target: {target}')

        img, target = self.prepare(img, target)
        
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target
    

    # def __getitem__(self, idx):
    #     img, target = super(CocoDetection, self).__getitem__(idx)
    #     image_id = self.ids[idx]
    #     anno = []
    #     for item in 
    #     target = {'image_id': image_id, 'annotations': target}

    #     print(f'Getting item {idx}, image_id {image_id}')
    #     print(f'Original target: {target}')

    #     img, target = self.prepare(img, target)
        
    #     if self._transforms is not None:
    #         img, target = self._transforms(img, target)
    #     return img, target

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

        duplicate_indices = get_box_dup(boxes)

        boxes = [item for idx, item in enumerate(boxes) if idx not in duplicate_indices]
        # guard against no boxes via resizing, convert (x,y,w,h)->(x,y,x,y)
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = [item for idx, item in enumerate(classes) if idx not in duplicate_indices]
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

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = [obj["area"] for obj in anno]
        area = [item for idx, item in enumerate(area) if idx not in duplicate_indices]
        area = torch.tensor(area)

        iscrowd = [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno]
        iscrowd = [item for idx, item in enumerate(iscrowd) if idx not in duplicate_indices]
        iscrowd = torch.tensor(iscrowd)
        
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


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

def imshow(img):
    img = img.numpy().transpose((1, 2, 0))  # chuyển từ tensor sang numpy
    img = np.clip(img, 0,1)
    # plt.imshow(img)
    # plt.axis('off')
    # print(type(img))
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

def build_data(image_set):
    root = Path(args.vgRoot)
    assert root.exists(), f'provided COCO path {root} does not exist'
    PATHS = {
        "train": (root / args.img_folder_coco, root / args.ann_path_coco / 'train_small_new.json'),
        "val": (root / args.img_folder_coco, root / args.ann_path_coco / 'val_small_new.json'),
    }

    print(PATHS[image_set])

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=False)
    return dataset

# def build_data(image_set):
#     root = Path('Datasets/mscoco')
#     assert root.exists(), f'provided COCO path {root} does not exist'
#     mode = 'instances'
#     PATHS = {
#         "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
#         "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
#     }

#     img_folder, ann_file = PATHS[image_set]
#     dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=False)
#     return dataset


if __name__ == '__main__':
    trainDataset = build_data('train')
    print(len(trainDataset))
    index = 50
    imageTransform, target = trainDataset.__getitem__(index)
    bboxes_scaled = rescale_bboxes(target['boxes'], (imageTransform.shape[2], imageTransform.shape[1]))
    # print(imageTransform.shape)
    # print(target)
    # print(bboxes_scaled)
    imCopy = imshow(imageTransform)
    imgPil = Image.fromarray((imCopy * 255).astype(np.uint8))
    plt.figure(figsize=(16,10))
    plt.imshow(imgPil)
    ax = plt.gca()
    for (xmin, ymin, xmax, ymax) in bboxes_scaled:
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color='red', linewidth=3))

    plt.show()
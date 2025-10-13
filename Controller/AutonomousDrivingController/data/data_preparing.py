import json
import os
from torch.utils.data import Dataset
import torch
from PIL import Image

class AutoData(Dataset):

    def __init__(self, driveable, line_lane, box, category, root_image, root_anno):

        self.root_image = root_image
        self.root_anno = root_anno

        with open(os.path.join(self.root_anno, driveable)) as drive:
            self.poly2d_drive = json.load(drive)

        with open(os.path.join(self.root_anno, line_lane)) as lane:
            self.poly2d_lane = json.load(lane)

        with open(os.path.join(self.root_anno, box)) as box:
            self.box2d = json.load(box)

    def __getitem__(self, index):
    
        image_id = self.box2d[index]['image_id']
        image_pth = os.path.join(self.root_image, image_id)
        img = Image.open(image_pth).convert("RGB")

        tgt_poly2d_drive = self.poly2d_drive[index]['labels']
        tgt_poly2d_lane = self.poly2d_lane[index]['labels']
        tgt_box2d = self.box2d[index]['labels']

class Preparing(object):

    def __init__(self):
        pass

    def __call__(self, img, tgt_poly2d_drive, tgt_poly2d_lane, tgt_box2d):

        w,h = img.size
        poly2d_drive = [obj['vertices'] for obj in tgt_poly2d_drive]
        poly2d_lane = [obj['vertices'] for obj in tgt_poly2d_lane]
        box2d = [obj['box2d'] for obj in tgt_box2d]

        target = {}

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])








        

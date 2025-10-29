import os
import torch
import config as args

root_image_folfer = args.ConfigDataBDD.image_folfer
root_anno_folder = args.ConfigDataBDD.anno_folder
root_image_seg_folder = args.ConfigDataBDD.image_seg_folder

train_driveable_anno =args.ConfigDataBDD.train_driveable_anno
train_lane_anno = args.ConfigDataBDD.train_lane_anno
train_box_anno = args.ConfigDataBDD.train_box_anno

valid_driveable_anno = args.ConfigDataBDD.valid_driveable_anno
valid_lane_anno = args.ConfigDataBDD.valid_lane_anno
valid_box_anno =args.ConfigDataBDD.valid_box_anno

catgories = args.ConfigDataBDD.categories

batch_size = 2
num_workers = 1
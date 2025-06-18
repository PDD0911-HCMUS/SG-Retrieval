import os
import torch
import config as args

log_dir = os.path.join(os.getcwd(), 'Controller/IRESGController/work_dir')
save_ckpt = os.path.join(os.getcwd(), 'Checkpoint', 'IRESG')

# Dataset
num_workers = 4
batch_size = 30
max_lenght = 10
max_triplet = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
tokenizer = "bert-base-uncased"
anno_train = args.ConfigData.iresg_train
anno_valid = args.ConfigData.iresg_valid
vg_image_dir = args.ConfigData.img_folder_vg

#Transformer encoder:
hidden_dim=256
nhead=8
nlayer=6
d_ffn=2048
dropout=0.1
activation="relu"

#Vision Encoder:
position_embedding='sine'
backbone='resnet50' # choose resnet50, resnet101, 
dilation=False
frozen_weights=None
lr_backbone=1e-05
masks=False

#Graph Encoder:
random_erasing_prob=0.5
pre_train = 'bert-base-uncased'

# Training
lr_drop=10
lr=0.0001
weight_decay=0.0001
epochs=40
start_epoch = 0
log_interval = 10
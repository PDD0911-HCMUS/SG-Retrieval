import os
import torch
import config as args

log_dir = os.path.join(os.getcwd(), 'Controller/HybridEncoderRegionDescriptionController/work_dir')
save_ckpt = os.path.join(os.getcwd(), 'Checkpoint', 'HybridEncoderRegionDescription')

# Dataset
num_workers = 0
batch_size = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
tokenizer = "bert-base-uncased"
anno_train = args.ConfigData.hybrid_encoder_train
anno_valid = args.ConfigData.hybrid_encoder_valid
vg_image_dir = args.ConfigData.img_folder_vg

#Vision Encoder:
position_embedding='sine'
backbone='resnet50' # choose resnet50, resnet101, 
dilation=False
frozen_weights=None
lr_backbone=1e-05
masks=False

#Transformer:
hidden_dim=256
nhead=8
nlayer=6
d_ffn=2048
dropout=0.1
activation="relu"
return_intermediate_dec=True
pre_norm=False

#Hybrid
num_queries = 100

set_cost_bbox = 5
set_cost_giou = 2
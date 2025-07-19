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
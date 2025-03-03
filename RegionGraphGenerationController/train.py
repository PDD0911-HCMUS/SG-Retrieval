from datasets.create_data import build_data
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
import util.misc as utils
from typing import Iterable
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
import torch

from model.ceatt import build_model

def train_engine(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int):
    

    for im, tgt in data_loader:
        im = im.to(device)
        tgt = [{k: v.to(device) for k, v in t.items()} for t in tgt]

        i,g = model(im, tgt)
        # print(src.size())

        break
    pass

if __name__ == "__main__":

    batch_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    tokenizer = "bert-base-uncased"

    hidden_dim=256
    
    #Vision encoder:
    position_embedding='sine'
    backbone='resnet50' # choose resnet50, resnet101, 
    dilation=False
    frozen_weights=None
    lr_backbone=1e-05
    masks=False
    nhead=8
    nlayer=6
    d_ffn=2048
    dropout=0.1
    activation="relu"

    pre_train = 'bert-base-uncased'

    lr_drop=200
    lr=0.0001
    weight_decay=0.0001

    epochs=300
    start_epoch = 0
    
    vg_image_dir = "/home/duypd/ThisPC-DuyPC/SG-Retrieval/0_Datasets/VisualGenome/VG_100K/"
    vg_anno_train = "/home/duypd/ThisPC-DuyPC/SG-Retrieval/0_Datasets/VisualGenome/anno_rg/train_data.json"
    vg_anno_val = "/home/duypd/ThisPC-DuyPC/SG-Retrieval/0_Datasets/VisualGenome/anno_rg/val_data.json"

    dataset_train = build_data(image_set = 'train',
                         annotation_file=vg_anno_train, 
                         image_folder=vg_image_dir, tokenizer = tokenizer)
    
    dataset_val = build_data(image_set = 'val',
                         annotation_file=vg_anno_val, 
                         image_folder=vg_image_dir, tokenizer = tokenizer)
    
    sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_val)
    batch_sampler_train = BatchSampler(sampler_train, batch_size, drop_last=True)
    

    data_train = DataLoader(dataset_train, 
                            batch_sampler=batch_sampler_train, 
                            collate_fn=utils.collate_fn
                        )
    
    data_val = DataLoader(dataset_val, 
                        batch_size=batch_size, 
                        sampler=sampler_val,
                        drop_last=False,
                        collate_fn=utils.collate_fn
                    )
    

    model = build_model(hidden_dim,lr_backbone,masks, backbone, dilation, 
                nhead, nlayer, d_ffn, dropout, activation, pre_train)
    
    criterion = None

    model_without_ddp = model

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": lr_backbone,
        },
    ]


    optimizer = torch.optim.AdamW(param_dicts, lr=lr,
                                  weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop)

    for epoch in range(start_epoch, epochs):
        train_engine(model, criterion, data_train, optimizer, device, epoch)
        break
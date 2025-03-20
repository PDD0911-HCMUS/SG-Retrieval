from Controller.CrossEncoderController.datasets.create_data import build_data
from Controller.CrossEncoderController.model.ceatt import build_model
from Controller.CrossEncoderController.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
import Controller.CrossEncoderController.util.misc as utils
from typing import Iterable
import torch
import config as args
import logging
import os
import time
import datetime
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch random seed
    torch.cuda.manual_seed(seed)  # Cho GPU
    torch.cuda.manual_seed_all(seed)  # If use multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensure fixed results for cuDNN
    torch.backends.cudnn.benchmark = False  # Turn off benchmarking to avoid differences between runs


def setup_logger(log_dir, log_file="LOGGER.log"):

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    
    # Delete old handlers if any (avoid log duplication)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Console handler (displayed on terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # File handler (write to file)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # Log format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def save_checkpoint(model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    epoch: int, 
                    losses, 
                    save_dir):
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"epoch_{epoch}.pth")
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": losses,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")

def train_engine(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, logger):
    
    model.train()
    criterion.train()

    total_loss = 0.0
    total_loss_v2r = 0.0
    total_loss_r2v = 0.0
    num_batches = len(data_loader)

    start_time = time.time()

    for batch_idx, (im, tgt) in enumerate(data_loader, start=1):
        batch_start_time = time.time()
        im = im.to(device)
        tgt = [{k: v.to(device) for k, v in t.items()} for t in tgt]

        vision,region = model(im, tgt)
        losses = criterion(vision,region)

        optimizer.zero_grad()
        losses['loss'].backward()

        # Gradient norm (helps control exploding gradient)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += losses['loss'].item()
        total_loss_v2r += losses['loss_v2r'].item()
        total_loss_r2v += losses['loss_r2v'].item()

        # ETA
        batch_time = time.time() - batch_start_time
        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / batch_idx) * num_batches
        eta = estimated_total_time - elapsed_time

        # Logging loss
        logger.info(
            f"Epoch {epoch} - Iter {batch_idx}/{num_batches} "
            f"- Time per batch: {batch_time:.2f}s "
            f"- ETA: {eta/60:.1f} min "
            f"- Loss_v2r = {losses['loss_v2r'].item():.4f} "
            f"- Loss_r2v = {losses['loss_r2v'].item():.4f} "
            f"- Loss = {losses['loss'].item():.4f} "
            f"- Grad Norm: {grad_norm:.4f}"
        )

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_loss_v2r = total_loss_v2r / num_batches if num_batches > 0 else 0
    avg_loss_r2v = total_loss_r2v / num_batches if num_batches > 0 else 0
    logger.info(f"Epoch {epoch} - Average Training Loss: {avg_loss:.4f}"
                f"- Loss_v2r: {avg_loss_v2r:.4f} "
                f"- Loss_r2v: {avg_loss_r2v:.4f}")
        
    return avg_loss

def valid_engine(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable,
                    device: torch.device, epoch: int, logger):
    
    model.eval()
    criterion.eval()

    total_loss = 0.0
    total_loss_v2r = 0.0
    total_loss_r2v = 0.0
    num_batches = len(data_loader)

    with torch.no_grad():
        for batch_idx, (im, tgt) in enumerate(data_loader, start=1):
            im = im.to(device)
            tgt = [{k: v.to(device) for k, v in t.items()} for t in tgt]
            vision,region = model(im, tgt)
            losses = criterion(vision,region)
            total_loss += losses['loss'].item()        
            total_loss_v2r += losses['loss_v2r'].item()
            total_loss_r2v += losses['loss_r2v'].item()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_loss_v2r = total_loss_v2r / num_batches if num_batches > 0 else 0
    avg_loss_r2v = total_loss_r2v / num_batches if num_batches > 0 else 0
    logger.info(
        f"Epoch {epoch} - Validation Loss: {avg_loss:.4f} "
        f"- Loss_v2r: {avg_loss_v2r:.4f} "
        f"- Loss_r2v: {avg_loss_r2v:.4f}"
    )
    return avg_loss

if __name__ == "__main__":

    set_seed(42)
    # Logger and save checkpoint
    log_dir = os.path.join(os.getcwd(), 'Controller/CrossEncoderController/work_dir')
    save_ckpt = os.path.join(os.getcwd(), 'Checkpoint', 'CrossEncoderAttention')
    logger = setup_logger(log_dir)

    # Dataset
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    tokenizer = "bert-base-uncased"
    vg_image_dir = args.ConfigData.img_folder_vg
    vg_anno_train = args.ConfigData.cross_encoder_train
    vg_anno_val = args.ConfigData.cross_encoder_valid

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

    #Region Encoder:
    random_erasing_prob=0.3
    pre_train = 'bert-base-uncased'

    # Training
    lr_drop=100
    lr=0.0001
    weight_decay=0.0001
    epochs=200
    start_epoch = 0

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
    

    model, criterion = build_model(hidden_dim,lr_backbone,masks, backbone, dilation, 
                nhead, nlayer, d_ffn, dropout, random_erasing_prob, activation, pre_train)

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

    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        losses_train = train_engine(model, criterion, data_train, optimizer, device, epoch, logger)
        lr_scheduler.step()
        save_checkpoint(model, optimizer, epoch, losses_train, save_ckpt)
        losses_valid = valid_engine(model, criterion, data_val, device, epoch, logger)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
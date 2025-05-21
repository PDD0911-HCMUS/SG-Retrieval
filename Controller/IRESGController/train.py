from Controller.IRESGController.datasets.data import build_data, collate_fn_dual_image
from Controller.IRESGController.model.model import build
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
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
    # torch.cuda.manual_seed_all(seed)  # If use multi-GPU
    # torch.backends.cudnn.deterministic = True  # Ensure fixed results for cuDNN
    # torch.backends.cudnn.benchmark = False  # Turn off benchmarking to avoid differences between runs

def setup_logger(log_dir):

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"LOGGER_{timestamp}.log"
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
                device: torch.device, epoch: int, logger, log_interval):
    model.train()
    criterion.train()

    total_loss = 0.0
    total_loss_contrastive = 0.0
    total_loss_consistency = 0.0
    num_batches = len(data_loader)

    start_time = time.time()

    for batch_idx, (im_a, im_b, trip_que, trip_rev) in enumerate(data_loader, start=1):
        batch_start_time = time.time()
        im_a = im_a.to(device)
        im_b = im_b.to(device)
        trip_que = [{k: v.to(device) for k, v in t.items()} for t in trip_que]
        trip_rev = [{k: v.to(device) for k, v in t.items()} for t in trip_rev]

        # print(trip_que)
        
        out_a, out_r_a, out_b, out_r_b = model(im_a,im_b,trip_que,trip_rev)
        losses = criterion(out_a, out_r_a, out_b, out_r_b)

        optimizer.zero_grad()
        losses['loss'].backward()

        # Gradient norm (helps control exploding gradient)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += losses['loss'].item()
        total_loss_contrastive += losses['loss_contrastive'].item()
        total_loss_consistency += losses['loss_consistency'].item()

        # ETA
        batch_time = time.time() - batch_start_time
        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / batch_idx) * num_batches
        eta = estimated_total_time - elapsed_time

        # Logging loss
        if batch_idx % log_interval == 0 or batch_idx == num_batches:
            logger.info(
                f"Epoch {epoch} - Iter {batch_idx}/{num_batches} "
                f"- Time per batch: {batch_time:.2f}s "
                f"- ETA: {eta/60:.1f} min "
                f"- loss_contrastive = {losses['loss_contrastive'].item():.4f} "
                f"- loss_consistency = {losses['loss_consistency'].item():.4f} "
                f"- Loss = {losses['loss'].item():.4f} "
                f"- Grad Norm: {grad_norm:.4f}"
            )

        break

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_loss_contrastive = total_loss_contrastive / num_batches if num_batches > 0 else 0
    avg_loss_consistency = total_loss_consistency / num_batches if num_batches > 0 else 0
    logger.info(f"Epoch {epoch} - Average Training Loss: {avg_loss}"
                f"- loss_contrastive: {avg_loss_contrastive} "
                f"- loss_consistency: {avg_loss_consistency}")
        
    return avg_loss

def valid_engine(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable,
                    device: torch.device, epoch: int, logger):
    
    model.eval()
    criterion.eval()

    total_loss = 0.0
    total_loss_contrastive = 0.0
    total_loss_consistency = 0.0
    num_batches = len(data_loader)

    with torch.no_grad():
        for batch_idx, (im_a, im_b, trip_que, trip_rev) in enumerate(data_loader, start=1):
            im_a = im_a.to(device)
            im_b = im_b.to(device)
            trip_que = [{k: v.to(device) for k, v in t.items()} for t in trip_que]
            trip_rev = [{k: v.to(device) for k, v in t.items()} for t in trip_rev]

            out_a, out_r_a, out_b, out_r_b = model(im_a,im_b,trip_que,trip_rev)
            losses = criterion(out_a, out_r_a, out_b, out_r_b)
            
            total_loss += losses['loss'].item()        
            total_loss_contrastive += losses['loss_contrastive'].item()
            total_loss_consistency += losses['loss_consistency'].item()

            break

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_loss_contrastive = total_loss_contrastive / num_batches if num_batches > 0 else 0
    avg_loss_consistency = total_loss_consistency / num_batches if num_batches > 0 else 0
    logger.info(
        f"Epoch {epoch} - Validation Loss: {avg_loss} "
        f"- loss_contrastive: {avg_loss_contrastive} "
        f"- loss_consistency: {avg_loss_consistency}"
    )
    return avg_loss

if __name__ == "__main__":

    set_seed(42)
    # Logger and save checkpoint
    log_dir = os.path.join(os.getcwd(), 'Controller/IRESGController/work_dir')
    save_ckpt = os.path.join(os.getcwd(), 'Checkpoint', 'IRESG')
    logger = setup_logger(log_dir)

    # Dataset
    num_workers = 4
    batch_size = 12
    max_lenght = 7
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
    log_interval = 50
    
    dataset_train = build_data(
        image_folder=vg_image_dir,
        ann_file=anno_train,
        tokenizer=tokenizer,
        max_length=max_lenght,
        image_set='train'
    )
    
    dataset_val = build_data(
        image_folder=vg_image_dir,
        ann_file=anno_valid,
        tokenizer=tokenizer,
        max_length=max_lenght,
        image_set='val'
    )

    sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_val)
    batch_sampler_train = BatchSampler(sampler_train, batch_size, drop_last=True)

    data_train = DataLoader(dataset_train, 
                            batch_sampler=batch_sampler_train, 
                            collate_fn=collate_fn_dual_image,
                            num_workers=num_workers,  # Load data song song
                            pin_memory=True
                        )
    
    data_val = DataLoader(dataset_val, 
                        batch_size=batch_size, 
                        sampler=sampler_val,
                        drop_last=False,
                        collate_fn=collate_fn_dual_image,
                        num_workers=num_workers,
                        pin_memory=True
                    )

    logger.info(f"Training DataLoader Info:")
    logger.info(f"Total Samples: {len(dataset_train)}")
    logger.info(f"Total Batches: {len(data_train)}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Num Workers: {num_workers}")
    logger.info(f"Pin Memory: {data_train.pin_memory}")

    logger.info(f"Validation DataLoader Info:")
    logger.info(f"Total Samples: {len(dataset_val)}")
    logger.info(f"Total Batches: {len(data_val)}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Num Workers: {num_workers}")
    logger.info(f"Pin Memory: {data_val.pin_memory}")

    check_data = dataset_train.__getitem__(0)

    print(check_data[0])

    model, criterion = build(hidden_dim,lr_backbone,masks, backbone, dilation, 
                nhead, nlayer, d_ffn, dropout, random_erasing_prob, activation, pre_train)
    
    model = model.to(device)
    criterion = criterion.to(device)

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
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5)

    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        losses_train = train_engine(model, criterion, data_train, optimizer, device, epoch, logger, log_interval)
        
        save_checkpoint(model, optimizer, epoch, losses_train, save_ckpt)
        losses_valid = valid_engine(model, criterion, data_val, device, epoch, logger)

        lr_scheduler.step(losses_valid)

        break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # split_json(anno)

    
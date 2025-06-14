from Controller.IRESGController.dataset.data import build_data, collate_fn_dual_image
from Controller.IRESGController.dataset.create_db import create_db, collate_fn_dual_image_db
from Controller.IRESGController.model.model import build, ModelCross
from Controller.IRESGController.util.misc import setup_logger
from Controller.IRESGController.train_engine import *
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
from typing import Iterable
import torch
import os
import time
import datetime
import random
import numpy as np
from config_run import *

def set_seed(seed=42):
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch random seed
    torch.cuda.manual_seed(seed)  # Cho GPU

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

if __name__ == "__main__":

    set_seed(42)

    logger = setup_logger(log_dir)
    
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

    dataset_db = create_db(
        image_folder=vg_image_dir,
        ann_file=anno_valid,
        tokenizer=tokenizer,
        max_length=max_lenght
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
    
    data_db = DataLoader(dataset_db,
                        batch_size=1, 
                        sampler=sampler_val,
                        drop_last=False,
                        collate_fn=collate_fn_dual_image_db,
                        num_workers=num_workers,
                        pin_memory=True)

    for name, dataset, loader in [
        ("Training", dataset_train, data_train),
        ("Validation", dataset_val, data_val)
    ]:
        logger.info(f"{name} DataLoader Info: "
                    f"Samples={len(dataset)}, "
                    f"Batches={len(loader)}, "
                    f"Batch Size={batch_size}, "
                    f"Workers={num_workers}, "
                    f"Pin Memory={loader.pin_memory}")

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
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop)

    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        losses_train = train_engine(model, criterion, data_train, optimizer, device, epoch, logger, log_interval)
        
        save_checkpoint(model, optimizer, epoch, losses_train, save_ckpt)
        losses_valid = valid_engine(model, criterion, data_val, data_db, device, epoch, logger, log_interval)

        lr_scheduler.step(losses_valid)

        break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # split_json(anno)

    
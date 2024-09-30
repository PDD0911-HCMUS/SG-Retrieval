from datasets.data import build_dataset
from model.obde import build_model
import os
import numpy as np
import random
from tqdm import tqdm
import argparse
import datetime
import time
from torch.utils.data import DataLoader
import torch
import util.misc as utils
from typing import Iterable

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--num_workers', default=2, type=int)

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--seq_length', type=int, default=7,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    return parser

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,):
    model.train()
    criterion.train()

    running_loss = 0.0  # Khởi tạo biến để lưu loss của epoch này
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch [{epoch}]")
    for batch_idx, (images, targets) in progress_bar:
        images = images.to(device)
        targets = [
            {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
            for t in targets
        ]

        optimizer.zero_grad()

        outputs = model(images)

        losses = criterion(outputs, targets)

        losses.backward()
        optimizer.step()

        running_loss += losses.item()

        # if batch_idx % 100 == 0:
        #     print(f"Epoch [{epoch}], Step [{batch_idx}/{len(data_loader)}], Loss: {losses.item():.4f}")

    avg_loss = running_loss / len(data_loader)
    print(f"Epoch [{epoch}] completed. Average Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, criterion, data_loader, device):
    model.eval() 
    criterion.eval()  

    running_loss = 0.0

    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluating")

    with torch.no_grad():
        for batch_idx, (images, targets) in progress_bar:
            images = images.to(device)
            targets = [
                {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
                for t in targets
            ]

            outputs = model(images)

            losses = criterion(outputs, targets)

            running_loss += losses.item()

            progress_bar.set_postfix(loss=losses.item())

    avg_loss = running_loss / len(data_loader)
    print(f"Evaluation completed. Average Loss: {avg_loss:.4f}")

    return avg_loss 

def main(args):
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device is being used: {device}')

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
 
    save_dir = '/radish/phamd/duypd-proj/SG-Retrieval/ckpt/objde/'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model, criterion = build_model(args)
    model.to(device)
    criterion.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train')
    dataset_val = build_dataset(image_set='val')

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    print("Start training")
    log_file = open(os.path.join(save_dir, "training_log.txt"), "w")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        avg_train_loss = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch)
        avg_val_loss = evaluate(model, criterion, data_loader_val, device)

        lr_scheduler.step() 
        log_file.write(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}\n")
        if (epoch + 1) % 2 == 0:
            model_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
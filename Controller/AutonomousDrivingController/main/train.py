import argparse
import datetime
import json
import math
from pathlib import Path
import sys
import time
import os

import torch
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler

from datasets import get_coco_api_from_dataset
from datasets.bdd import build
from trainer import Trainer
import util.misc as utils
from models.hyda import build_model
from models.criterion import build_criterion
import yaml


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def init_distributed_mode():
    """
    Tự động phát hiện xem có đang chạy DDP (torchrun) hay không.

    Trả về:
        distributed: bool
        rank: int
        world_size: int
        gpu: int (local_rank)
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(
            backend='nccl',
            init_method='env://'
        )
        torch.cuda.set_device(gpu)
        distributed = True
        print(f"[DDP] Initialized. rank={rank}, world_size={world_size}, gpu={gpu}")
    else:
        distributed = False
        rank = 0
        world_size = 1
        gpu = 0
        print("[DDP] Not using distributed mode (single process).")

    return distributed, rank, world_size, gpu


class Train:
    def __init__(self, model_yml, common_yml, criterion_yml):
        self.model_cfg = self._load_cfg(model_yml)
        self.common_cfg = self._load_cfg(common_yml)
        self.criterion_cfg = self._load_cfg(criterion_yml)

    def _load_cfg(self, yml_file):
        with open(yml_file, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg

    def _create_model(self):
        cfg = self.model_cfg['model']
        model, postprocessors = build_model(
            cfg['hidden_dim'],
            cfg['backbone']['position_embedding'],
            cfg['backbone']['lr_backbone'],
            cfg['backbone']['based'],
            cfg['backbone']['dilation'],
            cfg['backbone']['return_interm_layers'],
            cfg['transformer']['dropout'],
            cfg['transformer']['nheads'],
            cfg['transformer']['dim_feedforward'],
            cfg['transformer']['enc_layers'],
            cfg['transformer']['dec_layers'],
            cfg['transformer']['pre_norm'],
            cfg['num_queries'],
            cfg['num_drive_queries'],
            cfg['aux_loss'],
            cfg['num_classes'],
            cfg['training']
        )
        return model, postprocessors

    def _create_criterion(self):
        model_cfg, criterion_cfg = self.model_cfg['model'], self.criterion_cfg['coefficients']
        criterion = build_criterion(
            model_cfg['num_classes'],
            criterion_cfg['eos_coef'],
            criterion_cfg['bbox_loss_coef'],
            criterion_cfg['giou_loss_coef'],
            criterion_cfg['mask_loss_coef'],
            criterion_cfg['dice_loss_coef'],
            criterion_cfg['ahs_loss_coef'],
            model_cfg['aux_loss'],
            model_cfg['transformer']['dec_layers'],
            criterion_cfg['set_cost_class'],
            criterion_cfg['set_cost_bbox'],
            criterion_cfg['set_cost_giou']
        )
        return criterion

    def _build_data(self, distributed):
        common_cfg = self.common_cfg['common']
        batch_size = common_cfg['training']['batch_size']
        size = common_cfg['data']['size']

        dataset_train = build(
            "train",
            common_cfg['data']['root_image'],
            common_cfg['data']['root_anno'],
            common_cfg['data']['root_seg'],
            size
        )
        dataset_valid = build(
            "val",
            common_cfg['data']['root_image'],
            common_cfg['data']['root_anno'],
            common_cfg['data']['root_seg'],
            size
        )

        print(f"dataset_valid_length: {len(dataset_valid)}")
        print(f"dataset_train_length: {len(dataset_train)}")

        if distributed:
            # DDP: dùng DistributedSampler
            sampler_train = DistributedSampler(dataset_train, shuffle=True)
            sampler_val = DistributedSampler(dataset_valid, shuffle=False)

            data_loader_train = DataLoader(
                dataset_train,
                batch_size=batch_size,
                sampler=sampler_train,
                collate_fn=utils.collate_fn,
                num_workers=common_cfg['training']['num_workers'],
                pin_memory=True
            )

            data_loader_val = DataLoader(
                dataset_valid,
                batch_size=batch_size,
                sampler=sampler_val,
                drop_last=False,
                collate_fn=utils.collate_fn,
                num_workers=common_cfg['training']['num_workers'],
                pin_memory=True
            )
        else:
            # Single-GPU / single-process
            sampler_train = RandomSampler(dataset_train)
            sampler_val = SequentialSampler(dataset_valid)
            batch_sampler_train = BatchSampler(sampler_train, batch_size, drop_last=True)

            data_loader_train = DataLoader(
                dataset_train,
                batch_sampler=batch_sampler_train,
                collate_fn=utils.collate_fn,
                num_workers=common_cfg['training']['num_workers'],
                pin_memory=True
            )

            data_loader_val = DataLoader(
                dataset_valid,
                batch_size=batch_size,
                sampler=sampler_val,
                drop_last=False,
                collate_fn=utils.collate_fn,
                num_workers=common_cfg['training']['num_workers'],
                pin_memory=True
            )

        base_ds = get_coco_api_from_dataset(dataset_valid)
        return data_loader_train, data_loader_val, base_ds

    def load_checkpoint(self, model,
                        optimizer,
                        lr_scheduler,
                        resume_path,
                        device):
        """
        Load checkpoint và trả về:
            - start_epoch: epoch để tiếp tục train
            - output_dir: thư mục chứa checkpoint

        Args:
            model:        torch.nn.Module
            optimizer:    torch.optim.Optimizer (hoặc None)
            lr_scheduler: torch.optim.lr_scheduler._LRScheduler (hoặc None)
            resume_path:  str hoặc Path tới file .pth
            device:       torch.device hoặc str (ví dụ "cuda")

        Return:
            start_epoch (int), output_dir (Path hoặc None)
        """
        if resume_path is None or resume_path == "" or str(resume_path).lower() == "none":
            print("[Resume] No checkpoint path provided, training from scratch.")
            return 0, None

        resume_path = Path(resume_path)
        if not resume_path.is_file():
            print(f"[Resume] WARNING: Checkpoint file not found: {resume_path}")
            print("[Resume] Training from scratch.")
            return 0, None

        print(f"[Resume] Loading checkpoint from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")

        # --- Load model state ---
        state_dict = checkpoint.get("model", None)
        if state_dict is None:
            raise ValueError(f"[Resume] 'model' key not found in checkpoint: {resume_path}")

        model.load_state_dict(state_dict)
        model.to(device)
        print("[Resume] Model state loaded.")

        # --- Load optimizer state (nếu có) ---
        if optimizer is not None and "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                print("[Resume] Optimizer state loaded.")
            except Exception as e:
                print(f"[Resume] WARNING: Failed to load optimizer state: {e}")

        # --- Load lr_scheduler state (nếu có) ---
        if lr_scheduler is not None and "lr_scheduler" in checkpoint:
            try:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                print("[Resume] LR scheduler state loaded.")
            except Exception as e:
                print(f"[Resume] WARNING: Failed to load lr_scheduler state: {e}")

        # --- Xác định epoch bắt đầu ---
        if "epoch" in checkpoint:
            last_epoch = checkpoint["epoch"]
            start_epoch = last_epoch + 1
            print(f"[Resume] Last finished epoch in checkpoint: {last_epoch}")
            print(f"[Resume] Will continue from epoch: {start_epoch}")
        else:
            print("[Resume] 'epoch' not found in checkpoint, start_epoch = 0")
            start_epoch = 0

        output_dir = resume_path.parent  # thư mục chứa checkpoint

        return start_epoch, output_dir

    def run(self):
        # --- Init distributed (nếu có) ---
        distributed, rank, world_size, gpu = init_distributed_mode()

        # Device:
        if distributed:
            device = torch.device(f"cuda:{gpu}")
        else:
            device = torch.device(self.common_cfg['common']['device'])

        # --- Config resume & output_dir ---
        resume_path = self.common_cfg['common'].get('resume', None)

        output_dir = None
        log_file = None

        # build data (tuỳ theo distributed hay không)
        data_loader_train, data_loader_val, base_ds = self._build_data(distributed=distributed)

        # --- Model & Criterion ---
        model, postprocessors = self._create_model()
        criterion = self._create_criterion()

        model.to(device)
        criterion.to(device)

        model.train()
        criterion.train()

        # --- Optimizer & Scheduler ---
        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters()
                        if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters()
                           if "backbone" in n and p.requires_grad],
                "lr": self.model_cfg['model']['backbone']['lr_backbone'],
            },
        ]

        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=self.common_cfg['common']['training']['lr'],
            weight_decay=self.common_cfg['common']['training']['weight_decay']
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            self.common_cfg['common']['training']['lr_drop']
        )

        # --- Resume logic ---
        start_epoch_cfg = self.common_cfg['common']['start_epoch']
        start_epoch = start_epoch_cfg

        if resume_path:
            loaded_start_epoch, loaded_output_dir = self.load_checkpoint(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                resume_path=resume_path,
                device=device
            )
            start_epoch = max(start_epoch_cfg, loaded_start_epoch)
            if loaded_output_dir is not None:
                output_dir = loaded_output_dir

        # --- Wrap DDP (sau khi load checkpoint) ---
        if distributed:
            # DDP: model đã ở đúng device (cuda:gpu)
            model = DDP(model, device_ids=[gpu], output_device=gpu, find_unused_parameters=False)
            model_without_ddp = model.module
        else:
            model_without_ddp = model

        # --- Nếu chưa có output_dir (train mới) thì tạo mới ---
        if output_dir is None and self.common_cfg['common']['output_dir']:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.common_cfg['common']['output_dir']) / timestamp
            output_dir.mkdir(parents=True, exist_ok=True)

        # --- Logger (chỉ main process log ra file) ---
        if output_dir is not None and (not distributed or rank == 0):
            log_dir = output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

            log_fh = open(log_file, 'w')
            sys.stdout = Tee(sys.stdout, log_fh)
            sys.stderr = Tee(sys.stderr, log_fh)

            print(f"[Logger] Training logs will be saved to: {log_file}")
            print(f"[Logger] Starting training at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            if rank == 0:
                print("[Logger] No output_dir specified or not main process, logs only to console.")

        # --- Trainer ---
        start_time = time.time()

        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            postprocessors=postprocessors,
            base_ds=base_ds,
            output_dir=output_dir,
            max_norm=self.common_cfg['common']['training']['clip_max_norm']
        )

        # --- Training Loop ---
        for epoch in range(start_epoch, self.common_cfg['common']['epochs']):
            if distributed:
                # rất quan trọng cho DistributedSampler
                if isinstance(data_loader_train.sampler, DistributedSampler):
                    data_loader_train.sampler.set_epoch(epoch)

            train_stats = trainer.train_one_epoch(
                data_loader_train,
                self.common_cfg['common']['training']['print_freq_train'],
                epoch
            )
            lr_scheduler.step()

            if output_dir is not None:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                if (epoch + 1) % 2 == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                    }, checkpoint_path)

            # Evaluate (chỉ cần coco_evaluator ở main process; Trainer bên trong có thể đã xử lý)
            test_stats, coco_evaluator = trainer.evaluate(data_loader_val, self.common_cfg['common']['training']['print_freq_val'])

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if output_dir is not None and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 2 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                output_dir / "eval" / name
                            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if rank == 0:
            print('Training time {}'.format(total_time_str))

        # --- Cleanup DDP ---
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    model_yml = "configs/model_cfg.yaml"
    common_yml = "configs/common_cfg.yaml"
    criterion_yml = "configs/criterion_cfg.yaml"

    train = Train(model_yml, common_yml, criterion_yml)
    train.run()
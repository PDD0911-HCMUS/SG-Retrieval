import argparse
import datetime
import json
import math
from pathlib import Path
import sys
import time
import torch
import numpy as np
from datasets import get_coco_api_from_dataset
from datasets.bdd import build
from trainer import Trainer
import util.misc as utils
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
from models.detr import build_model
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

class Train:
    def __init__(self, model_yml, common_yml, criterion_yml):
        self.model_cfg =  self._load_cfg(model_yml)
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
            cfg['aux_loss'],
            cfg['num_classes']
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
            model_cfg['aux_loss'],
            model_cfg['transformer']['dec_layers'],
            criterion_cfg['set_cost_class'],
            criterion_cfg['set_cost_bbox'],
            criterion_cfg['set_cost_giou']
        )
        return criterion
    
    def _build_data(self):
        common_cfg = self.common_cfg['common']
        batch_size = common_cfg['training']['batch_size']
        size = common_cfg['data']['size']
        dataset_train = build("train", 
                              common_cfg['data']['root_image'],
                              common_cfg['data']['root_anno'],
                              common_cfg['data']['root_seg'],
                              size)
        dataset_valid = build("val",
                              common_cfg['data']['root_image'],
                              common_cfg['data']['root_anno'],
                              common_cfg['data']['root_seg'],
                              size)
        print(f"dataset_valid_length: {dataset_valid.__len__()}")
        print(f"dataset_train_length: {dataset_train.__len__()}")
        sampler_train = RandomSampler(dataset_train)
        sampler_val = SequentialSampler(dataset_valid)
        batch_sampler_train = BatchSampler(sampler_train, batch_size, drop_last=True)
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=common_cfg['training']['num_workers'])
    
        data_loader_val = DataLoader(dataset_valid, batch_size, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn, num_workers=common_cfg['training']['num_workers'])
        base_ds = get_coco_api_from_dataset(dataset_valid)

        return data_loader_train, data_loader_val, base_ds
    
    def run(self):
        device = torch.device(self.common_cfg['common']['device'])

        if self.common_cfg['common']['output_dir']:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.common_cfg['common']['output_dir']) / timestamp
            output_dir.mkdir(parents=True, exist_ok=True)

            log_dir = Path(output_dir) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        data_loader_train, data_loader_val, base_ds = self._build_data()
        
        model, postprocessors = self._create_model()
        criterion = self._create_criterion()
        model.to(device)
        criterion.to(device)
        
        model.train()
        criterion.train()

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.model_cfg['model']['backbone']['lr_backbone'],
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, 
                                      lr=self.common_cfg['common']['training']['lr'],
                                      weight_decay=self.common_cfg['common']['training']['weight_decay'])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                       self.common_cfg['common']['training']['lr_drop'])
        
        print("Start training")
        log_fh = open(log_file, 'w')
        sys.stdout = Tee(sys.stdout, log_fh)
        sys.stderr = Tee(sys.stderr, log_fh)

        print(f"[Logger] Training logs will be saved to: {log_file}")
        print(f"[Logger] Starting training at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

        for epoch in range(self.common_cfg['common']['start_epoch'], self.common_cfg['common']['epochs']):
            train_stats = trainer.train_one_epoch(
                data_loader_train, 
                self.common_cfg['common']['training']['print_freq_train'], 
                epoch)
            lr_scheduler.step()

            if self.common_cfg['common']['output_dir']:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                if (epoch + 1) % 2 == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        # 'args': args,
                    }, checkpoint_path)

            test_stats, coco_evaluator = trainer.evaluate(data_loader_val)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
            
            if self.common_cfg['common']['output_dir'] and utils.is_main_process():
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
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    output_dir / "eval" / name)
            break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))   
        print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    model_yml = "Controller/AutonomousDrivingController/config/model_cfg.yaml"
    common_yml = "Controller/AutonomousDrivingController/config/common_cfg.yaml"
    criterion_yml = "Controller/AutonomousDrivingController/config/criterion_cfg.yaml"

    train = Train(model_yml, common_yml, criterion_yml)
    train.run()

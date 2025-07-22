from config_run import *
from Controller.HybridEncoderRegionDescriptionController.datasets.data import build_data
from Controller.HybridEncoderRegionDescriptionController.models.hybrid_encoder import build
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
import util.misc as utils
import json

if __name__ == "__main__":

    dataset_train = build_data(
        image_folder=vg_image_dir,
        anno_file=anno_train,
        tokenizer=tokenizer,
        image_set='train'
    )

    dataset_val = build_data(
        image_folder=vg_image_dir,
        anno_file=anno_valid,
        tokenizer=tokenizer,
        image_set='val'
    )

    sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_val)
    batch_sampler_train = BatchSampler(sampler_train, batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    
    model = build(hidden_dim, num_queries,
          lr_backbone,masks, backbone, dilation, # Vision Encoder
          dropout,nhead,d_ffn,nlayer,activation,pre_norm, return_intermediate_dec, # Transformer Module
          set_cost_bbox, set_cost_giou # Matcher
          )
    model = model.to(device)

    print(dataset_train.__len__())
    print(dataset_val.__len__())

    ignored_keys = {'regions', 'image_id'}
    for samples, targets in data_loader_train:
        print(targets)
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k not in ignored_keys} for t in targets]
        src = model(samples, targets)
        break
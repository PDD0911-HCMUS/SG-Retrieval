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
    
    model = build(hidden_dim, lr_backbone,masks, backbone, dilation, num_queries)
    model = model.to(device)

    print(dataset_train.__len__())
    print(dataset_val.__len__())

    for img, tgt in data_loader_train:
        print(tgt)

        src = model(img)
        break
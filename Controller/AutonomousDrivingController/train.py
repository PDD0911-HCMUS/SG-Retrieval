from Controller.AutonomousDrivingController.data.data_preparing import build
from config_run import *
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
from tqdm import tqdm


if __name__=="__main__":
    dataset_train = build(image_set="train")

    dataset_valid = build(image_set="val")

    sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_valid)
    batch_sampler_train = BatchSampler(sampler_train, batch_size, drop_last=True)

    print(dataset_train.__len__())
    print(dataset_valid.__len__())

    img, tgt = dataset_train.__getitem__(0)
    print(f"image input size: {img.size()}")
    print(f"target info: {tgt}")
    print(f"masks size: {tgt['masks'].size()}")
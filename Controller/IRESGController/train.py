from Controller.IRESGController.datasets.data import build_data


import random
import numpy as np
import torch
import config as args
import json

def set_seed(seed=42):
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch random seed
    torch.cuda.manual_seed(seed)  # Cho GPU
    # torch.cuda.manual_seed_all(seed)  # If use multi-GPU
    # torch.backends.cudnn.deterministic = True  # Ensure fixed results for cuDNN
    # torch.backends.cudnn.benchmark = False  # Turn off benchmarking to avoid differences between runs

def split_json(input_file, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):
    # 1. Đọc dữ liệu
    with open(input_file, 'r') as f:
        data = json.load(f)

    # 2. Shuffle để đảm bảo ngẫu nhiên
    random.seed(seed)
    random.shuffle(data)

    # 3. Tính số lượng mẫu cho mỗi phần
    total = len(data)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    # 4. Chia dữ liệu
    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]

    # 5. Ghi ra file
    with open('train.json', 'w') as f:
        json.dump(train_data, f, indent=4)

    with open('valid.json', 'w') as f:
        json.dump(valid_data, f, indent=4)

    with open('test.json', 'w') as f:
        json.dump(test_data, f, indent=4)

    print(f"✅ Split complete: train({len(train_data)}), valid({len(valid_data)}), test({len(test_data)})")

if __name__ == "__main__":
    set_seed(42)

    # Dataset
    num_workers = 0
    batch_size = 16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    tokenizer = "bert-base-uncased"
    anno = args.ConfigData.iresg_anno
    image_folder = args.ConfigData.img_folder_vg

    print(anno)

    with open(anno, 'r') as f:
        data = json.load(f)

    print(len(data))

    # split_json(anno)

    
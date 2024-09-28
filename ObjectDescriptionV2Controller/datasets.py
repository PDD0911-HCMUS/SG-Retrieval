import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize image to 256x256
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
])

class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, json_folder, image_folder, data_name, split, transform=None):
        """
        :param json_folder: folder where JSON files are stored
        :param image_folder: folder where images are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Load image paths and encoded captions from JSON file
        with open(os.path.join(json_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.data = json.load(j)  # Load image_id, image_path, and captions

        # Load caption lengths from JSON file
        with open(os.path.join(json_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Store the image folder path
        self.image_folder = image_folder

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Captions per image (assuming all images have the same number of captions)
        self.cpi = len(self.data[0]['captions'])

        # Total number of datapoints
        self.dataset_size = len(self.data) * self.cpi

    def __getitem__(self, i):
        # Determine the image and caption index
        img_idx = i // self.cpi  # The index of the image
        cap_idx = i % self.cpi   # The index of the caption for that image

        # Load the image from the correct folder using image_path
        img_path = os.path.join(self.image_folder, os.path.basename(self.data[img_idx]['image_path']))
        img = Image.open(img_path).convert("RGB")  # Convert to RGB if necessary

        # Apply transformation to the image
        if self.transform is not None:
            img = self.transform(img)

        # Get the corresponding caption and its length
        caption = torch.LongTensor(self.data[img_idx]['captions'][cap_idx])
        caplen = torch.LongTensor([self.caplens[i]])

        if self.split == 'TRAIN':
            return img, caption, caplen
        else:
            # For validation or testing, return all captions for the image for BLEU-4 scoring
            all_captions = torch.LongTensor(self.data[img_idx]['captions'])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
    

# # Giả sử bạn đã có các file JSON và ảnh trong hai thư mục khác nhau
# json_folder = '/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/VisualGenome/anno_org'  # Thư mục chứa các file JSON
# image_folder = '/home/duypd/ThisPC-DuyPC/SG-Retrieval/Datasets/VisualGenome/VG_100K_cropped'  # Thư mục chứa các file ảnh
# data_name = 'coco'                      # Tên của tập dữ liệu, ví dụ 'coco'
# split = 'TRAIN'                         # Chọn split, có thể là 'TRAIN', 'VAL', hoặc 'TEST'

# # Khởi tạo dataset
# dataset = CaptionDataset(json_folder=json_folder, image_folder=image_folder, data_name=data_name, split=split, transform=transform)

# # Khởi tạo DataLoader để tạo các batch từ dataset
# batch_size = 8  # Số lượng mẫu trong mỗi batch
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# # Vòng lặp để duyệt qua dữ liệu
# for i, (images, captions, caplens) in enumerate(dataloader):
#     print(f'Batch {i + 1}')
#     print('Images:', images.shape)     # Kích thước batch ảnh
#     print('Captions:', captions.shape) # Kích thước batch captions
#     print('Caption lengths:', caplens) # Độ dài của từng chú thích trong batch

#     # Dừng lại sau khi duyệt qua vài batch
#     if i == 2:
#         break

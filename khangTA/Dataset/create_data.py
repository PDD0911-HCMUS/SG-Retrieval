import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from transformers import AutoTokenizer
import unicodedata

classes = {
    'Công văn': 0,
    'Quyết định': 1,
    'Báo cáo': 2,
    'Thông báo': 3,
    'Tờ trình': 4,
    'Thư mời': 5,
    'Đơn': 6,
    'Giấy mời': 7
}

tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')

class CreateData(Dataset):
    def __init__(self, csv_file):
        self.csv_file = csv_file

        self.data_frame = pd.read_csv(csv_file, encoding='utf-8')

    def __getitem__(self, index):
        input_text = self.data_frame.loc[index, 'TRICH_YEU']
        label = self.data_frame.loc[index, 'HINHTHUC']

        input_tok, imput_msk, label = transforms_data(input_text, label)
        return input_tok, imput_msk, label
    
    def __len__(self):
        return len(self.data_frame)
    

def transforms_data(input_text, label):
    input_text = input_text.lower()
    # input_text = unicodedata.normalize('NFD', input_text)
    # input_text = ''.join(c for c in input_text if unicodedata.category(c) != 'Mn')
    token = tokenizer(input_text, padding='max_length', truncation=True, max_length=32, return_tensors="pt")

    input_tok = token['input_ids'][0]
    imput_msk = token['attention_mask'][0]

    label = torch.as_tensor(classes[label])

    return input_tok, imput_msk, label

def build_data(csv_file, ratio):
    dataset = CreateData(csv_file=csv_file)
    train_size = int(ratio * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    return train_dataset, valid_dataset
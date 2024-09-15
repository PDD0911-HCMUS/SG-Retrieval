import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from collections import Counter
import nltk

nltk_data_dir = os.path.expanduser('/radish/phamd/duypd-proj/SG-Retrieval/nltk_data')
nltk.data.path.append(nltk_data_dir)
nltk.download('punkt_tab', download_dir=nltk_data_dir) 

from torch.nn.utils.rnn import pad_sequence
import json

path_to_images = 'Datasets/VisualGenome/VG_100K_cropped/'

with open('Datasets/VisualGenome/anno_org/caption.json', 'r') as f:
    data_anno = json.load(f)
annotations_vocab = data_anno['annotations']

with open('Datasets/VisualGenome/anno_org/captions_object_train.json', 'r') as f:
    data_train = json.load(f)
images_train = data_train['images']
annotations_train = data_train['annotations']

with open('Datasets/VisualGenome/anno_org/captions_object_val.json', 'r') as f:
    data_val = json.load(f)
images_val = data_val['images']
annotations_val = data_val['annotations']

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Augmentation
    transforms.RandomRotation(10),  # Augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def numericalize_caption(caption, vocab):
    numericalized_caption = [vocab.stoi["<SOS>"]]  # Thêm <SOS> ở đầu
    numericalized_caption += vocab.numericalize(caption)
    numericalized_caption.append(vocab.stoi["<EOS>"])  # Thêm <EOS> ở cuối
    return numericalized_caption

class CaptionDataset(Dataset):
    def __init__(self, images, annotations, vocab, transform=None):
        self.images = images
        self.annotations = annotations
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_id = annotation['image_id']
        caption = annotation['caption']
        
        # Lấy đường dẫn ảnh
        image_info = next(img for img in self.images if img['id'] == image_id)
        image_path = f"{path_to_images}/{image_info['file_name']}"
        image = Image.open(image_path).convert("RGB")
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Chuyển caption thành số
        numericalized_caption = numericalize_caption(caption, self.vocab)
        
        return image, torch.tensor(numericalized_caption)

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    
    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenize(text):
        return nltk.word_tokenize(text.lower())
    
    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [self.stoi.get(word, self.stoi["<UNK>"]) for word in tokenized_text]


vocab = Vocabulary(freq_threshold=5)
vocab.build_vocabulary([anno['caption'] for anno in annotations_vocab])

def collate_fn(batch):
    images = []
    captions = []
    
    for img, caption in batch:
        images.append(img)
        captions.append(caption)
    
    images = torch.stack(images, dim=0)
    captions = pad_sequence(captions, batch_first=True, padding_value=vocab.stoi["<PAD>"])
    
    return images, captions

def build_data():
    
    train_dataset = CaptionDataset(images_train, annotations_train, vocab, transform=train_transform)
    val_dataset = CaptionDataset(images_val, annotations_val, vocab, transform=val_transform)

    return train_dataset, val_dataset
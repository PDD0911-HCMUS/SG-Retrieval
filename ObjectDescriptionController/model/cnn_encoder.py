import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class CNN_Encoder(nn.Module):
    def __init__(self, embed_size, dropout=0.5):
        super(CNN_Encoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Bỏ lớp FC và Adaptive AvgPool
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Pooling để giảm kích thước đặc trưng

    def forward(self, images):
        features = self.resnet(images)  # [batch_size, 2048, H, W]
        features = self.adaptive_pool(features)  # [batch_size, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # [batch_size, 2048]
        features = self.dropout(self.fc(features))  # [batch_size, embed_size]
        return features

def build_encoder(embed_size):
    cnn_encoder = CNN_Encoder(embed_size)
    return cnn_encoder
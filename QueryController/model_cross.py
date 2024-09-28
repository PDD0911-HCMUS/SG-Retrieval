import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer

# Text Encoder và Label Encoder sử dụng BERT
class Encoder(nn.Module):
    def __init__(self, output_dim):
        super(Encoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']  # Truy cập pooler_output từ outputs
        return self.fc(pooled_output)

class CrossModalModel(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossModalModel, self).__init__()
        self.text_encoder = Encoder(embed_dim)
        self.label_encoder = Encoder(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, input_ids, attention_mask, label_ids, label_attention_mask):
        text_features = self.text_encoder(input_ids, attention_mask)
        label_features = self.label_encoder(label_ids, label_attention_mask)
        
        # Áp dụng cơ chế attention
        text_features = text_features.unsqueeze(0)  # Thêm chiều batch
        label_features = label_features.unsqueeze(0)  # Thêm chiều batch
        
        attn_output, _ = self.attention(text_features, label_features, label_features)
        attn_output = attn_output.squeeze(0)  # Loại bỏ chiều batch
        
        # Kết hợp đầu ra attention với embedding gốc
        combined_features = text_features.squeeze(0) + attn_output
        combined_features = self.fc(combined_features)
        
        return combined_features, label_features.squeeze(0)

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity()

    def forward(self, text_features, label_features):
        positive_similarity = self.cosine_similarity(text_features, label_features)
        negative_similarity = self.cosine_similarity(text_features, torch.roll(label_features, shifts=1, dims=0))
        
        loss = torch.mean(torch.clamp(self.margin - positive_similarity + negative_similarity, min=0))
        return loss
    

def build_model(device):
    embed_dim = 256
    num_heads = 8
    model = CrossModalModel(embed_dim, num_heads).to(device)
    criterion = ContrastiveLoss().to(device)

    return model, criterion
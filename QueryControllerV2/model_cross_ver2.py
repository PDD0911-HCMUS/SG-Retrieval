import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertModel, BertTokenizer
import timm

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim):
        super(ImageEncoder, self).__init__()
        
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.reset_classifier(0)
        self.fc = nn.Linear(self.vit.embed_dim, embed_dim)

    def forward(self, images):
        features = self.vit(images)  
        return self.fc(features)


# Text Encoder và Label Encoder sử dụng BERT
class BertEncoder(nn.Module):
    def __init__(self, output_dim):
        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output'] 
        return self.fc(pooled_output)

class CrossModalModel(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossModalModel, self).__init__()
        self.im_encoder = ImageEncoder(embed_dim)
        self.trip_encoder = BertEncoder(embed_dim)
        
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, images, trip_ids, trip_attention_mask):
        im_features = self.im_encoder(images)
        trip_features = self.trip_encoder(trip_ids, trip_attention_mask)
        
        # Áp dụng cơ chế attention
        im_features = im_features.unsqueeze(0)
        trip_features = trip_features.unsqueeze(0)
        
        cross1, _ = self.attention(im_features, trip_features, trip_features)
        cross1 = cross1.squeeze(0) 

        cross2, _ = self.attention(trip_features, im_features, im_features)
        cross2 = cross2.squeeze(0) 
        
        # # Kết hợp đầu ra attention với embedding gốc
        # combined_features = text_features.squeeze(0) + attn_output
        # combined_features = self.fc(combined_features)
        
        return cross1, cross2
        # return combined_features, label_features.squeeze(0)

# Contrastive Loss
class CrossLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(CrossLoss, self).__init__()
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity()

    def forward(self, cross1, cross2):
        positive_similarity = self.cosine_similarity(cross1, cross2)
        negative_similarity = self.cosine_similarity(cross1, torch.roll(cross2, shifts=1, dims=0))
        
        loss = torch.mean(torch.clamp(self.margin - positive_similarity + negative_similarity, min=0))
        return loss
    
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, image_embeddings, trip_embeddings):
        # Normalize the embeddings
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        trip_embeddings = F.normalize(trip_embeddings, p=2, dim=-1)

        # Compute the logits (dot product of image and text embeddings)
        logits_per_image = torch.matmul(image_embeddings, trip_embeddings.t()) / self.temperature
        logits_per_trip = logits_per_image.t()

        # Labels are the indices of the diagonal
        labels = torch.arange(logits_per_image.size(0)).long().to(image_embeddings.device)

        # Compute the cross-entropy loss for both image-to-text and text-to-image
        loss_image_to_trip = F.cross_entropy(logits_per_image, labels)
        loss_trip_to_image = F.cross_entropy(logits_per_trip, labels)

        # Return the average of both losses
        return (loss_image_to_trip + loss_trip_to_image) / 2

def build_model(device):
    embed_dim = 256
    num_heads = 8
    model = CrossModalModel(embed_dim, num_heads).to(device)
    criterion = InfoNCELoss().to(device)

    return model, criterion
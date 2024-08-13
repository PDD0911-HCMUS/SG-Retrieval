import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
from RelTR.models.backbone import Backbone, Joiner
from RelTR.models.position_encoding import PositionEmbeddingSine
from RelTR.models.transformer import Transformer
from RelTR.models.reltr import RelTR
import ConfigArgs as args
import os
import json

def project_embeddings_2(embeddings, num_projection_layers, projection_dims, dropout_rate):
    projection_layer = nn.Linear(embeddings.size(1), projection_dims).to(args.device)
    norm_layer = nn.LayerNorm(projection_dims).to(args.device)
    projected_embeddings = projection_layer(embeddings)
    for _ in range(num_projection_layers):
        x = F.relu(projected_embeddings)
        x = nn.Linear(projection_dims, projection_dims).to(args.device)(x)
        x = F.dropout(x, p=dropout_rate)
        x = projected_embeddings + x
        projected_embeddings = norm_layer(x)
    return projected_embeddings

class ProjectEmbeddings(nn.Module):
    def __init__(self, input_dims, projection_dims, num_projection_layers, dropout_rate):
        super(ProjectEmbeddings, self).__init__()
        self.projection_dims = projection_dims
        self.num_projection_layers = num_projection_layers
        self.dropout_rate = dropout_rate

        self.initial_projection = nn.Linear(input_dims, projection_dims)
        
        self.projection_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(projection_dims, projection_dims),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for _ in range(num_projection_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(projection_dims)

    def forward(self, embeddings):
        # First projection from input dimension to projection dimension
        x = self.initial_projection(embeddings)
        
        # Apply each projection layer defined in the module list
        for layer in self.projection_layers:
            residual = x
            x = layer(x)
            x = x + residual  # Add skip connection
        
        # Apply layer normalization
        x = self.layer_norm(x)
        return x
    
class TextEncoder(nn.Module):
    def __init__(self, input_dims, num_projection_layers, projection_dims, dropout_rate):
        super(TextEncoder, self).__init__()
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased').to(args.device)
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')

        self.num_projection_layers = num_projection_layers
        self.projection_dims = projection_dims
        self.dropout_rate = dropout_rate
        self.input_dims = input_dims

    def forward(self, x):
        outputs = self.bert(x)
        pooled_output = outputs['pooler_output']
        # x = project_embeddings(pooled_output, self.num_projection_layers, self.projection_dims, self.dropout_rate)
        # tx_norm = self.project_embeddings(pooled_output)
        tx_norm = project_embeddings_2(pooled_output, self.num_projection_layers, 
                                                    self.projection_dims, 
                                                    self.dropout_rate)
        return tx_norm
    
class RelTREncoder(nn.Module):
    def __init__(self, input_dims, num_projection_layers, projection_dims, dropout_rate):
        super(RelTREncoder, self).__init__()
        position_embedding = PositionEmbeddingSine(128, normalize=True)
        backbone = Backbone('resnet50', False, False, False)
        backbone = Joiner(backbone, position_embedding)
        backbone.num_channels = 2048

        transformer = Transformer(d_model=256, dropout=0.1, nhead=8, 
                                dim_feedforward=2048,
                                num_encoder_layers=6,
                                num_decoder_layers=6,
                                normalize_before=False,
                                return_intermediate_dec=True)

        self.reltr = RelTR(backbone, transformer, num_classes=151, num_rel_classes = 51,
                    num_entities=100, num_triplets=200)

        # The checkpoint is pretrained on Visual Genome
        ckpt = torch.hub.load_state_dict_from_url(
            url='https://cloud.tnt.uni-hannover.de/index.php/s/PB8xTKspKZF7fyK/download/checkpoint0149.pth',
            map_location='cpu', check_hash=True)
        
        self.reltr.load_state_dict(ckpt['model'])

        self.input_dims = input_dims
        self.num_projection_layers = num_projection_layers
        self.projection_dims = projection_dims
        self.dropout_rate = dropout_rate

    def forward(self, x):
        _, sg_map = self.reltr(x)
        sg_map = torch.max(sg_map, dim=1, keepdim=True).values
        sg_map = torch.squeeze(sg_map, dim=1)
        sg_norm = project_embeddings_2(sg_map, self.num_projection_layers, 
                                                    self.projection_dims, 
                                                    self.dropout_rate)
        return sg_norm

class SGGnBert(nn.Module):
    def __init__(self, reltr, bert):
        super().__init__()
        self.reltr = reltr
        self.bert = bert
    def forward(self, x_img, x_txt_em):
        sg_norm = self.reltr(x_img)
        tx_norm = self.bert(x_txt_em)
        return sg_norm, tx_norm

class DotProductSimilarityLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(DotProductSimilarityLoss, self).__init__()
        self.temperature = temperature
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, queries_embeddings, sgg_embeddings):
        # Tính dot-product similarity
        logits = torch.matmul(queries_embeddings, sgg_embeddings.t()) / self.temperature

        # Tính self-similarity cho captions và images
        queries_similarity = torch.matmul(queries_embeddings, queries_embeddings.t()) / self.temperature
        
        sgg_similarity = torch.matmul(sgg_embeddings, sgg_embeddings.t()) / self.temperature

        # Tính targets dựa trên sự tương đồng của captions và images
        targets_similarity = (queries_similarity + sgg_similarity) / 2

        # Sử dụng softmax để chuyển đổi similarity thành xác suất
        targets = F.softmax(targets_similarity, dim=1)

        # Tính cross-entropy loss
        loss = self.cross_entropy_loss(logits, targets.argmax(dim=1))
        return loss
    
# class DotProductSimilarityLoss(nn.Module):
#     def __init__(self, temperature=1.0):
#         super(DotProductSimilarityLoss, self).__init__()
#         self.temperature = temperature
#         self.cross_entropy_loss = nn.CrossEntropyLoss()

#     def forward(self, queries_embeddings, sgg_embeddings):

#         print(f'queries_embeddings {queries_embeddings.size()}')
#         print(f'sgg_embeddings {sgg_embeddings.size()}')
#         # Tính dot-product similarity
#         logits = torch.matmul(queries_embeddings, sgg_embeddings.t()) / self.temperature

#         # Tạo nhãn đích cho mỗi ví dụ là chính nó
#         targets = torch.arange(logits.size(0)).type_as(logits).long()

#         # Tính cross-entropy loss
#         loss = self.cross_entropy_loss(logits, targets)
#         return loss
    
def build_text_encoder(input_dims, num_projection_layers, projection_dims, dropout_rate):
    txt_en = TextEncoder(input_dims, num_projection_layers, projection_dims, dropout_rate)
    return txt_en

def build_sgg_encoder(input_dims, num_projection_layers, projection_dims, dropout_rate):
    sgg_en = RelTREncoder(input_dims, num_projection_layers, projection_dims, dropout_rate)
    return sgg_en

    
def build_model(num_projection_layers, projection_dims, dropout_rate):
    input_dims_sgg = 512
    input_dims_txt = 256
    txt_en = build_text_encoder(input_dims_txt, num_projection_layers, projection_dims, dropout_rate)
    sgg_en = build_sgg_encoder(input_dims_sgg, num_projection_layers, projection_dims, dropout_rate)
    dual_model = SGGnBert(sgg_en, txt_en)
    dot_loss = DotProductSimilarityLoss(temperature=1)
    return dual_model, dot_loss
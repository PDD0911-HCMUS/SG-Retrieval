import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .res_backbone import build_backbone
from .vision_encoder import build_vision_encoder

class RGG(nn.Module):
    def __init__(self, vision_encoder, region_decoder, hidden_dim, num_entities, num_regions):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.region_decoder = region_decoder

        self.num_regions = nn.Embedding(num_regions, hidden_dim)
        self.num_entities = nn.Embedding(num_entities, hidden_dim)
    
    def forward(self, img: NestedTensor):

        vision, vision_msk, vision_pos = self.vision_encoder(img)

        reg = self.region_decoder(
            entity = self.num_entities.weight,
            region = self.num_regions.weight,
            memory = vision,
            memory_key_padding_mask=vision_msk
        )

        return vision
    
class Criterion(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, vision_embed, region_embed):
        vision_emb = F.normalize(vision_embed, dim=1)
        graph_emb = F.normalize(region_embed, dim=1)

        # Tính ma trận similarity: [B, B]
        logits = torch.matmul(vision_emb, graph_emb.t()) / self.temperature

        labels = torch.arange(logits.size(0), device=vision_emb.device)

        loss_v2r = F.cross_entropy(logits, labels)
        loss_r2v = F.cross_entropy(logits.t(), labels)

        loss = (loss_v2r + loss_r2v) / 2
        return loss

def build_model(hidden_dim,lr_backbone,masks, backbone, dilation, 
                nhead, nlayer, d_ffn, dropout, activation, num_entities, num_regions):

    vision_backbone = build_backbone(hidden_dim,lr_backbone,masks, backbone, dilation)
    vision_encoder = build_vision_encoder(vision_backbone, hidden_dim, nhead, nlayer, d_ffn, dropout, activation)
    
    model = RGG(vision_encoder, hidden_dim, num_entities, num_regions)

    return model
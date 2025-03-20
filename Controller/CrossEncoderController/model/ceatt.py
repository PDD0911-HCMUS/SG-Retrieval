from Controller.CrossEncoderController.util import box_ops
from Controller.CrossEncoderController.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

import torch
import torch.nn.functional as F
from torch import nn
from .res_backbone import build_backbone
from .vision_encoder import build_vision_encoder
from .region_graph_encoder import build_graph_encoder

class CEAtt(nn.Module):
    def __init__(self, vision_encoder, graph_encoder, hidden_dim, nhead, dropout):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.graph_encoder = graph_encoder

        self.attn_vision = nn.MultiheadAttention(hidden_dim, nhead, dropout, batch_first=True)
        self.attn_graph = nn.MultiheadAttention(hidden_dim, nhead, dropout, batch_first=True)
    
    def forward(self, img: NestedTensor, tgt):

        vision, vision_msk, _ = self.vision_encoder(img)

        region, region_msk = self.graph_encoder(tgt)

        vision, _ = self.attn_vision(
            query=vision,
            key=region,
            value=region,
            key_padding_mask=region_msk  # mask cho graph
        )

        region, _ = self.attn_graph(
            query=region,
            key=vision,
            value=vision,
            key_padding_mask=vision_msk  # mask cho vision
        )

        print(vision.size())
        print(region.size())
        print(vision[:, 0].size(),region[:,0].size())

        return vision[:, 0],region[:,0]
    
class Criterion(nn.Module):
    def __init__(self, temperature=0.03):
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

        losses = {
            "loss_v2r": loss_v2r,
            "loss_r2v": loss_r2v,
            "loss": (loss_v2r + loss_r2v) / 2
        }

        return losses

def build_model(hidden_dim,lr_backbone,masks, backbone, dilation, 
                nhead, nlayer, d_ffn, dropout, random_erasing_prob, activation, pre_train):

    vision_backbone = build_backbone(hidden_dim,lr_backbone,masks, backbone, dilation)
    vision_encoder = build_vision_encoder(vision_backbone, hidden_dim, nhead, nlayer, d_ffn, dropout, activation)
    graph_encoder = build_graph_encoder(hidden_dim, nhead, nlayer, d_ffn, dropout, random_erasing_prob, activation, pre_train)
    
    model = CEAtt(vision_encoder, graph_encoder, hidden_dim, nhead, dropout)

    criterion = Criterion(temperature=0.07)

    return model, criterion
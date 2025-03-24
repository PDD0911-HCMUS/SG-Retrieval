from .res_backbone import build_backbone
import torch
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
import copy

class VisionEncoder(nn.Module):
    def __init__(self, backbone, hidden_dim, nhead=8, nlayer=6, d_ffn=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.backbone = backbone
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        self.encoder_layer = TransformerEncoderLayer(hidden_dim, nhead, d_ffn, dropout, activation)
        self.layers = _get_clones(self.encoder_layer, nlayer)

        self.nlayer = nlayer
        self.hidden_dim = hidden_dim
        self.nhead = nhead

    def forward(self, img: NestedTensor):
        if isinstance(img, (list, torch.Tensor)):
            img = nested_tensor_from_tensor_list(img)
        features, pos = self.backbone(img)

        src, mask = features[-1].decompose()
        assert mask is not None

        output = self.input_proj(src).flatten(start_dim=2).transpose(1,2)
        mask = mask.flatten(start_dim=1)
        pos = pos[-1].flatten(start_dim=2).transpose(1,2)

        for layer in self.layers:
            output = layer(output, src_key_padding_mask=mask, pos=pos)

        return output, mask, pos


class TransformerEncoderLayer(nn.Module):

    def __init__(self, hidden_dim, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, 
                src, 
                src_key_padding_mask: Optional[Tensor] = None, 
                pos: Optional[Tensor] = None):
        
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_vision_encoder(backbone, hidden_dim, nhead=8, nlayer=6, d_ffn=2048, dropout=0.1, activation="relu"):
    vision_encoder = VisionEncoder(backbone, hidden_dim, nhead, nlayer, d_ffn, dropout, activation)
    return vision_encoder
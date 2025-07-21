# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from transformers import BertTokenizer


class HybridEncoder(nn.Module):
    def __init__(self, backbone, transformer, matcher, # Module
                 num_queries, hidden_dim):
        
        super().__init__()

        self.num_queries = num_queries
        
        self.backbone = backbone
        self.transformer = transformer
        self.matcher = matcher

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

    def forward(self, samples: NestedTensor, targets):
        
        # Vision Encoder Line
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        src = self.input_proj(src)
        # End of Vision Encoder

        # Transformer Line
        # Encoder Block uses the Vision Encoder output for self-attention (src, mask, pos[-1])
        # Feed query_embed (learnable) to Decoder Block
        # Return hs [B, N_Queries, Hidden_Dim], maps [B, N_Queries, 1, H, W] ~ src [B, Hidden_Dim, H, W]
        
        hs, memory , maps = self.transformer(src, mask, self.query_embed.weight, pos[-1])

        map = maps[-1].squeeze(2)
        B, Nq, H, W = map.shape
        _, C, Hm, Wm = memory.shape

        assert H == Hm and W == Wm, "maps and memory must have the same spatial size"

        # Reshape
        map_flat = map.view(B, Nq, H * W) # [B, Nq, HW]
        map_flat = torch.softmax(map_flat, dim=-1) # normalize attention
        memory_flat = memory.view(B, C, H * W) # [B, C, HW]
        # Compute region_feat: [B, Nq, C]
        region_feat = torch.einsum('bnl,bcl->bnc', map_flat, memory_flat)

        # Bounding Boxes Output [B, N_Queries, 4]
        outputs_boxes= self.bbox_embed(hs).sigmoid()

        out = {'pred_boxes': outputs_boxes[-1]}
        indices = self.matcher(out, targets)
        batch_idx, query_idx = self._get_src_permutation_idx(indices)
        region_feat = region_feat[batch_idx, query_idx]  # [sum(N_gt), C]

        print(indices)
        print(f"Vision Encoder size: {src.size()}")
        print(f"Output Memory Transformer size: {memory.size()}")
        print(f"Region Feat size: {region_feat.size()}")
        print(f"Ouput Transformer size: {hs[-1].size()} and {hs.size()}")
        print(f"Output Maps Transformer size: {maps[-1].size()} and {maps.size()}")
        print(f"Ouput Boxes size: {outputs_boxes[-1].size()} and {outputs_boxes.size()}")
        return src
    
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([
            torch.full_like(src_idx, i)
            for i, (src_idx, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src_idx for (src_idx, _) in indices])
        return batch_idx, src_idx

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(hidden_dim, num_queries,
          lr_backbone,masks, backbone, dilation, # Vision Encoder
          dropout,nhead,d_ffn,nlayer,activation,pre_norm, return_intermediate_dec, # Transformer Module
          set_cost_bbox, set_cost_giou # Matcher
          ):

    vision_backbone = build_backbone(hidden_dim,lr_backbone,masks, backbone, dilation)
    transformer = build_transformer(hidden_dim, dropout, nhead, d_ffn, nlayer, activation, pre_norm, return_intermediate_dec)
    matcher = build_matcher(set_cost_bbox, set_cost_giou)

    model = HybridEncoder(
        vision_backbone,
        transformer,
        matcher,
        num_queries, 
        hidden_dim
    )

    return model
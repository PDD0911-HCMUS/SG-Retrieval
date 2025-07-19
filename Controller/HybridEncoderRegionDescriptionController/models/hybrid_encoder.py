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
    def __init__(self, backbone, num_queries, hidden_dim):
        
        super().__init__()

        self.num_queries = num_queries
        
        self.backbone = backbone

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

    def forward(self, samples: NestedTensor):
        
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()

        src = self.input_proj(src)

        print(src.size())
        return src

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


def build(hidden_dim, lr_backbone,masks, backbone, dilation, num_queries):

    vision_backbone = build_backbone(hidden_dim,lr_backbone,masks, backbone, dilation)

    model = HybridEncoder(
        vision_backbone,
        num_queries, 
        hidden_dim
    )

    return model
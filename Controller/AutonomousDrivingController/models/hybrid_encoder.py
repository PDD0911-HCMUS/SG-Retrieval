import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

class RepBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.act   = nn.SiLU(inplace=True)

    def forward(self, x):
        out = self.bn3(self.conv3(x)) + self.bn1(self.conv1(x)) + x
        return self.act(out)

class SemanticFusion(nn.Module):
    """
    Fusion block cho 2 feature maps S5 & S4 (nhánh detection trong HyDA-DETR).
    - S5 có độ phân giải nhỏ hơn, giàu ngữ nghĩa.
    - S4 có độ phân giải lớn hơn, nhiều chi tiết hình học.
    - Tự động downsample S4 -> size của S5 rồi concat theo kênh.
    - Hai nhánh song song: 1x1Conv và 1x1Conv + RepBlock (phiên bản nhẹ hơn SpatialFusion).
    - Element-wise add và kích hoạt SiLU.
    - Nếu return_flatten=True: trả về [H5*W5, B, C] (F5 trong sơ đồ).
    """
    def __init__(self, hidden_dim: int, num_rep_blocks: int = 1, return_flatten: bool = False):
        super().__init__()
        C = hidden_dim
        self.return_flatten = return_flatten

        # Sau concat: [B, 2C, H5, W5] -> [B, C, H5, W5]
        self.branch1_proj = nn.Conv2d(2*C, C, kernel_size=1, bias=False)
        self.branch1_bn   = nn.BatchNorm2d(C)

        self.branch2_proj = nn.Conv2d(2*C, C, kernel_size=1, bias=False)
        self.branch2_bn   = nn.BatchNorm2d(C)
        reps = [RepBlock(C) for _ in range(num_rep_blocks)]
        self.branch2_rep  = nn.Sequential(*reps)

        self.act = nn.SiLU(inplace=True)

    def forward(self, s5: torch.Tensor, s4: torch.Tensor):
        """
        s5: [B, C, H5, W5]
        s4: [B, C, H4, W4] (downsample -> [B, C, H5, W5])
        """
        B, C, H5, W5 = s5.shape

        # 1) Align S4 -> S5 (downsample)
        s4_down = F.interpolate(s4, size=(H5, W5), mode='bilinear', align_corners=False)

        # 2) Concat hai đặc trưng
        x = torch.cat([s5, s4_down], dim=1)  # [B, 2C, H5, W5]

        # 3) Hai nhánh song song
        b1 = self.branch1_bn(self.branch1_proj(x))          # [B, C, H5, W5]
        b2 = self.branch2_bn(self.branch2_proj(x))          # [B, C, H5, W5]
        b2 = self.branch2_rep(b2)

        # 4) Element-wise add + kích hoạt
        fused = self.act(b1 + b2)                           # [B, C, H5, W5]

        # 5) Flatten (nếu cần cho encoder input)
        if self.return_flatten:
            fused = fused.flatten(2).permute(2, 0, 1)       # [H5*W5, B, C]

        return fused
class SpatialFusion(nn.Module):
    """
    Fusion block (Figure 5 RT-DETR) cho 2 feature maps S3 & S4.
    - hidden_dim = d (ví dụ 256) sau input_proj.
    - Tự upsample S4 -> size của S3 rồi concat theo kênh.
    - Hai nhánh song song: 1x1Conv và 1x1Conv + RepBlock.
    - Element-wise add để ra tensor [B, d, Hs3, Ws3].
    - Nếu return_flatten=True: trả về [Hs3*Ws3, B, d] (F trong hình).
    """
    def __init__(self, hidden_dim: int, num_rep_blocks: int = 1, return_flatten: bool = False):
        super().__init__()
        C = hidden_dim
        self.return_flatten = return_flatten

        # Sau concat kênh: [B, 2C, H, W] -> đưa về C
        self.branch1_proj = nn.Conv2d(2*C, C, kernel_size=1, bias=False)
        self.branch1_bn   = nn.BatchNorm2d(C)

        self.branch2_proj = nn.Conv2d(2*C, C, kernel_size=1, bias=False)
        self.branch2_bn   = nn.BatchNorm2d(C)
        reps = [RepBlock(C) for _ in range(num_rep_blocks)]
        self.branch2_rep  = nn.Sequential(*reps)

        self.act = nn.SiLU(inplace=True)

    def forward(self, s3: torch.Tensor, s4: torch.Tensor):
        """
        s3: [B, C, H3, W3]
        s4: [B, C, H4, W4]  (upsample -> [B, C, H3, W3])
        """
        B, C, H3, W3 = s3.shape

        # 1) Align S4 -> S3
        s4_up = F.interpolate(s4, size=(H3, W3), mode='bilinear', align_corners=False)

        # 2) Concat
        x = torch.cat([s3, s4_up], dim=1)  # [B, 2C, H3, W3]

        # 3) Parallel Brach
        b1 = self.branch1_bn(self.branch1_proj(x))           # [B, C, H3, W3]
        b2 = self.branch2_bn(self.branch2_proj(x))           # [B, C, H3, W3]
        b2 = self.branch2_rep(b2)                            # [B, C, H3, W3]

        # 4) Element-wise add (+) and activate
        fused = self.act(b1 + b2)                            # [B, C, H3, W3]

        # 5) Flatten (F) [Option: Bool] for decoder: [L, B, C]
        if self.return_flatten:
            fused = fused.flatten(2).permute(2, 0, 1)        # [H3*W3, B, C]

        return fused

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output
    
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class HybridTransformerEncoder(nn.Module):
    
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_flatten=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        self.sp_fusion = SpatialFusion(hidden_dim=d_model, num_rep_blocks=1, return_flatten=return_flatten)
        self.sm_fusion = SemanticFusion(hidden_dim=d_model, num_rep_blocks=1, return_flatten=return_flatten)
        
    def forward(self, src, src3, src4, mask, pos_embed):
        
        # flatten NxCxHxW to HWxNxC
        # bs, c, h, w = src.shape
        src5 = self.sm_fusion(src, src4)
        #Encoder Forward
        memory = self.encoder(src5, src_key_padding_mask=mask, pos=pos_embed)
        #Fusion Spatail Feature
        sp_src = self.sp_fusion(src3, src4)

        return memory, sp_src

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


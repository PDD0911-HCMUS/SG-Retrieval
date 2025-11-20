from typing import List
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4,
                      context_dim // 8, context_dim // 16, context_dim // 64]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = self.out_lay(x)
        return x


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DriveSeg(nn.Module):
    """
    Driveable segmentation head với Query Soft Voting Aggregation (QSVA).

    Inputs:
        hs_sp:    [L, B, Q, C]  (không dùng trực tiếp trong logic dưới, nhưng giữ để tương thích)
        hm:       [L, B, nH, Q, HW]  (attention maps từ decoder)
        src3:     [B, C, Henc, Wenc] (spatial feature map)
        src3_proj:[B, C', Henc, Wenc] (projected feature cho mask_head)
        features: FPN features (list of NestedTensor); dùng features[i].tensors

    Output:
        seg_masks: [B, 1, H_out, W_out]
    """

    def __init__(self, hidden_dim, nheads):
        super().__init__()
        # mask conditioning từ heatmap (2-channel) lên thành vector 128-dim cho mỗi query
        self.so_mask_conv = nn.Sequential(
            nn.Upsample(size=(28, 28)),
            nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
        )
        self.so_mask_fc = nn.Sequential(
            nn.Linear(8192, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
        )

        # scoring mỗi query (lấy trọng số mềm alpha_i)
        self.score_mlp = MLP(128, 128, 1, 2)

        # mask head: input channels = hidden_dim (src3_proj) + nheads (attention)
        self.mask_head = MaskHeadSmallConv(
            hidden_dim + nheads, [1024, 512, 256], hidden_dim
        )

    def forward(self, hs_sp, hm, src3, src3_proj, features):
        """
        hm: [L, B, nH, Q, HW]  (ở đây nH thường = số head attention, ví dụ 2)
        """

        L, B, nH, Q, HW = hm.shape
        Henc, Wenc = src3.shape[-2], src3.shape[-1]
        assert HW == Henc * Wenc, "HW of hm must match Henc*Wenc of src3"

        # ------------------------------------------------------------------
        # 1) Chuẩn bị feature h cho từng query để scoring
        #    hm.view -> [-1, 2, Henc, Wenc] vì nH=2 (2-head attention maps)
        # ------------------------------------------------------------------
        # shape: [L*B*Q, 2, Henc, Wenc]
        h_conv_in = hm.view(-1, 2, Henc, Wenc)
        # shape: [L*B*Q, 32, h', w'] -> flatten -> [L, B, Q, -1]
        h = self.so_mask_conv(h_conv_in).view(
            hs_sp.shape[0], hs_sp.shape[1], hs_sp.shape[2], -1
        )  # [L, B, Q, 8192]
        # FC -> [L, B, Q, 128], lấy layer cuối L-1 -> [B, Q, 128]
        h = self.so_mask_fc(h)[-1]  # [B, Q, 128]

        # ------------------------------------------------------------------
        # 2) Query scoring -> trọng số mềm alpha cho từng query
        # ------------------------------------------------------------------
        scores = self.score_mlp(h).squeeze(-1)  # [B, Q]
        alpha = scores.softmax(dim=-1)          # [B, Q]

        # ------------------------------------------------------------------
        # 3) Lấy heatmap layer cuối & reshape về [B, nH, Q, H, W]
        # ------------------------------------------------------------------
        hm_last = hm[-1]                        # [B, nH, Q, HW]
        B2, nH2, Q2, HW2 = hm_last.shape
        assert B2 == B and nH2 == nH and Q2 == Q and HW2 == HW

        hm2d = hm_last.view(B, nH, Q, Henc, Wenc)   # [B, nH, Q, H, W]

        # ------------------------------------------------------------------
        # 4) Query Soft Voting Aggregation (QSVA)
        #    agg_map[b, h, x, y] = Σ_i alpha[b, i] * hm2d[b, h, i, x, y]
        # ------------------------------------------------------------------
        alpha_map = alpha.view(B, 1, Q, 1, 1)       # [B, 1, Q, 1, 1]
        agg_map = (alpha_map * hm2d).sum(dim=2)     # [B, nH, H, W]

        # seg-head requires [B, 1, nH, H, W] (Q=1 in front of nH):
        att_for_head = agg_map.unsqueeze(1)         # [B, 1, nH, H, W]

        seg_masks = self.mask_head(
            src3_proj,
            att_for_head,
            [features[2].tensors, features[1].tensors, features[0].tensors],
        )  # [B*1, 1, H_out, W_out] (batch đã flatten trong mask_head)

        return seg_masks.view(B, 1, seg_masks.shape[-2], seg_masks.shape[-1])
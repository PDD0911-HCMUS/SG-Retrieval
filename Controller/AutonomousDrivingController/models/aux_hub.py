from torch import Tensor, nn

class AuxHubHead(nn.Module):
    """
    Auxiliary Hub Supervision Head đặt trên S4:
    - Input:  S4 [B, C, H4, W4]
    - Output:
        aux_logits: [B, H4*W4, num_classes + 1]
        aux_boxes:  [B, H4*W4, 4]  (cx, cy, w, h, normalized 0–1)
    """
    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, s4: Tensor):
        """
        s4: [B, C, H4, W4]
        """
        B, C, H, W = s4.shape
        x = self.proj(s4) # [B, hidden_dim, H, W]
        x = x.flatten(2) # [B, hidden_dim, H*W]
        x = x.transpose(1, 2) # [B, H*W, hidden_dim]

        aux_logits = self.cls_head(x) # [B, N4, num_classes]
        aux_boxes  = self.box_head(x) # [B, N4, 4]
        aux_boxes  = aux_boxes.sigmoid() # normalize [0, 1]

        return aux_logits, aux_boxes

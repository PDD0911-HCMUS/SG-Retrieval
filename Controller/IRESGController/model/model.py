from Controller.IRESGController.util.misc import NestedTensor
from Controller.IRESGController.model.models import build_model
import torch
import torch.nn.functional as F
from torch import nn

class ModelCross(nn.Module):
    def __init__(self, model_a, model_b):
        super().__init__()

        self.model_a = model_a
        self.model_b = model_b

    def forward(self, img_a: NestedTensor, img_b: NestedTensor, tgt_a, tgt_b):

        out_a = self.model_a(img_a, tgt_a)
        out_b = self.model_a(img_b, tgt_b)
        
        print(out_a == out_b)

        return out_a, out_b

class Criterion(nn.Module):
    def __init__(self, temperature=0.03):
        super().__init__()
        self.temperature = temperature

    def forward(self, out_a, out_b):
        a = F.normalize(out_a, dim=1)
        b = F.normalize(out_b, dim=1)

        # Tính ma trận similarity: [B, B]
        logits = torch.matmul(a, b.t()) / self.temperature

        labels = torch.arange(logits.size(0), device=a.device)

        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.t(), labels)

        losses = {
            "loss_v2r": loss_a,
            "loss_r2v": loss_b,
            "loss": (loss_a + loss_b) / 2
        }

        return losses
    
def build(hidden_dim,lr_backbone,masks, backbone, dilation, 
        nhead, nlayer, d_ffn, dropout, random_erasing_prob, activation, pre_train):

    criterion = Criterion(temperature=0.07)

    model_a = build_model(hidden_dim,lr_backbone,masks, backbone, dilation, 
                nhead, nlayer, d_ffn, dropout, random_erasing_prob, activation, pre_train)
    
    model_b = build_model(hidden_dim,lr_backbone,masks, backbone, dilation, 
                nhead, nlayer, d_ffn, dropout, random_erasing_prob, activation, pre_train)
    
    model = ModelCross(model_a, model_b)

    return model, criterion
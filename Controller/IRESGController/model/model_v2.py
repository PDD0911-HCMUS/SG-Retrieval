from Controller.IRESGController.util.misc import NestedTensor
from Controller.IRESGController.model.models_v2 import build_model
import torch
import torch.nn.functional as F
from torch import nn

class ModelCross(nn.Module):
    def __init__(self, models):
        super().__init__()

        self.models = models

    def forward(self, img_a: NestedTensor, img_b: NestedTensor, tgt_a, tgt_b):

        z_iA, z_iB, zt_e, z_o = self.models(img_a, img_b, tgt_a, tgt_b)

        return z_iA, z_iB, zt_e, z_o

class Criterion(nn.Module):
    def __init__(self, temperature=0.07, alpha=1.0, margin=0.2):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # trọng số cho consistency loss
        self.margin = margin

    def compute_infonce(self, src, tgt):
        """
        query: [B, D] - z_o or zt_e (query features)
        key:   [B, D] - z_iB (gallery features)
        """
        # Normalize vectors
        src = F.normalize(src, p=2, dim=1)
        tgt   = F.normalize(tgt, p=2, dim=1)

        # Compute cosine similarity matrix: [B, B]
        logits = torch.matmul(src, tgt.T) / self.temperature

        # Ground-truth: diagonal is positive
        labels = torch.arange(src.size(0)).to(src.device)

        loss = F.cross_entropy(logits, labels)
        return loss
    
    def cosine_sim_loss(self, src, tgt):
        src = F.normalize(src, p=2, dim=1)
        tgt = F.normalize(tgt, p=2, dim=1)

        sim = (src * tgt).sum(dim=1)

        loss_sim = (1-sim).mean()
        return loss_sim


    def forward(self, z_iA, z_iB, zt_e, z_o):

        loss_o = self.compute_infonce(z_o, z_iB)
        loss_e = self.compute_infonce(zt_e, z_iB)

        cosine = (loss_o + loss_e) / 2

        return {
            "elem_sim": [loss_o, loss_e],
            "sim": cosine,
            "loss": cosine
        }
    
def build(hidden_dim,lr_backbone,masks, backbone, dilation, 
        nhead, nlayer, d_ffn, dropout, random_erasing_prob, activation, pre_train):

    criterion = Criterion(temperature=0.07)

    models = build_model(hidden_dim,lr_backbone,masks, backbone, dilation, 
                nhead, nlayer, d_ffn, dropout, random_erasing_prob, activation, pre_train)

    model = ModelCross(models)

    return model, criterion
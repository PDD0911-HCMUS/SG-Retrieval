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

        z_e_a = self.model_a(img_a, tgt_a)
        z_e_b = self.model_b(img_b, tgt_b)
        
        # print(out_a == out_b)

        return z_e_a, \
                z_e_b

class Criterion(nn.Module):
    def __init__(self, temperature=0.03, alpha=1.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # trọng số cho consistency loss

    def compute_contrastive_loss(self, z_e_a, z_e_b):
        z_e_a = F.normalize(z_e_a, dim=1)
        z_e_b = F.normalize(z_e_b, dim=1)

        logits = torch.matmul(z_e_a, z_e_b.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=z_e_a.device)

        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.t(), labels)
        return (loss_a + loss_b) / 2

    def compute_consistency_loss(self, z_e, z_r):
        z_e = F.normalize(z_e, dim=1)
        z_r = F.normalize(z_r, dim=1)

        cos = (z_e * z_r).sum(dim=1)        # [B]
        return (1 - cos).mean()
        # return F.mse_loss(z_e, z_r)

    def info_nce_loss(self, z_o, z_e):
        sim_matrix = torch.mm(z_o, z_e.t()) / self.temperature
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        loss_i2j = F.cross_entropy(sim_matrix, labels)
        loss_j2i = F.cross_entropy(sim_matrix.T, labels)
        return (loss_i2j + loss_j2i) / 2
    
    def cosine_sim_loss(self, z_i, z_o, z_e):
        z_i_norm = F.normalize(z_i, p=2, dim=1)
        z_o_norm = F.normalize(z_o, p=2, dim=1)
        z_e_norm = F.normalize(z_e, p=2, dim=1)

        sim_o = (z_i_norm * z_o_norm).sum(dim=1)
        sim_e = (z_i_norm * z_e_norm).sum(dim=1)

        loss_o = (1-sim_o).mean()
        loss_e = (1-sim_e).mean()
        return loss_o, loss_e


    def forward(self, z_i, z_o, z_e):
        info_nce = self.info_nce_loss(z_o, z_e)
        cosine_sim_o,  cosine_sim_e= self.compute_consistency_loss(z_i, z_o, z_e)

        total = info_nce + self.alpha*((cosine_sim_o + cosine_sim_e) / 2)
        return {
            "info_nce": info_nce,
            "cosine_sim": [cosine_sim_o,  cosine_sim_e],
            "loss": total
        }
    
def build(hidden_dim,lr_backbone,masks, backbone, dilation, 
        nhead, nlayer, d_ffn, dropout, random_erasing_prob, activation, pre_train):

    criterion = Criterion(temperature=0.07)

    model_a = build_model(hidden_dim,lr_backbone,masks, backbone, dilation, 
                nhead, nlayer, d_ffn, dropout, random_erasing_prob, activation, pre_train)
    
    model_b = build_model(hidden_dim,lr_backbone,masks, backbone, dilation, 
                nhead, nlayer, d_ffn, dropout, random_erasing_prob, activation, pre_train)
    
    model = ModelCross(model_a, model_b)

    return model, criterion
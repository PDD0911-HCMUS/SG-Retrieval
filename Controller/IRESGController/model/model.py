from Controller.IRESGController.util.misc import NestedTensor
from Controller.IRESGController.model.models import build_model
import torch
import torch.nn.functional as F
from torch import nn

class ModelCross(nn.Module):
    def __init__(self, models):
        super().__init__()

        self.models = models

    def forward(self, img_a: NestedTensor, img_b: NestedTensor, tgt_a, tgt_b):

        z_i, z_o, z_e, z_be = self.models(img_a, img_b, tgt_a, tgt_b)

        return z_o, z_e, z_i, z_be

class Criterion(nn.Module):
    def __init__(self, temperature=0.03, alpha=1.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # trọng số cho consistency loss

    def info_nce_loss(self, z_o, z_e):
        z_o = F.normalize(z_o, dim=1)
        z_e = F.normalize(z_e, dim=1)
        sim_matrix = torch.mm(z_o, z_e.t()) / self.temperature
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        loss_i2j = F.cross_entropy(sim_matrix, labels)
        loss_j2i = F.cross_entropy(sim_matrix.T, labels)
        return (loss_i2j + loss_j2i) / 2
    
    def cosine_sim_loss(self, z_i, z_t):
        z_i_norm = F.normalize(z_i, p=2, dim=1)
        z_t_norm = F.normalize(z_t, p=2, dim=1)

        sim = (z_i_norm * z_t_norm).sum(dim=1)

        loss_sim = (1-sim).mean()
        return loss_sim


    def forward(self, z_i, z_o, z_e, z_be):
        info_nce = self.info_nce_loss(z_o, z_e)
        info_nce_oeB = self.info_nce_loss(z_o, z_be)
        info_nce_eeB = self.info_nce_loss(z_e, z_be)

        cosine_sim_o = self.cosine_sim_loss(z_i, z_o)
        cosine_sim_e = self.cosine_sim_loss(z_i, z_e)
        cosine_sim_be = self.cosine_sim_loss(z_be, z_e)

        avg_cosine = (cosine_sim_o + cosine_sim_e + cosine_sim_be) / 3

        total = info_nce + self.alpha*avg_cosine

        return {
            "info_nce": info_nce,
            "cosine_sim": [cosine_sim_o,  cosine_sim_e, cosine_sim_be],
            "avg_cosine": avg_cosine,
            "loss": total
        }
    
def build(hidden_dim,lr_backbone,masks, backbone, dilation, 
        nhead, nlayer, d_ffn, dropout, random_erasing_prob, activation, pre_train):

    criterion = Criterion(temperature=0.07)

    models = build_model(hidden_dim,lr_backbone,masks, backbone, dilation, 
                nhead, nlayer, d_ffn, dropout, random_erasing_prob, activation, pre_train)
    
    # model_b = build_model(hidden_dim,lr_backbone,masks, backbone, dilation, 
    #             nhead, nlayer, d_ffn, dropout, random_erasing_prob, activation, pre_train)
    
    model = ModelCross(models)

    return model, criterion
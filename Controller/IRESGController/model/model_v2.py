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

        z_iA, z_iB, zt_e, z_o, z_eB_i2g = self.models(img_a, img_b, tgt_a, tgt_b)

        # print(z_iA.size(), z_o.size(), z_e.size(), z_eB.size())

        return z_iA, z_iB, zt_e, z_o, z_eB_i2g

class Criterion(nn.Module):
    def __init__(self, temperature=0.07, alpha=1.0, margin=0.2):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # trọng số cho consistency loss
        self.margin = margin

    def info_nce_loss(self, z_o, z_iB):
        
        z_o = F.normalize(z_o, dim=1)
        z_iB = F.normalize(z_iB, dim=1)
        
        sim_matrix = torch.mm(z_o, z_iB.t()) #/ self.temperature
        batch_size = sim_matrix.size(0)
        labels = torch.arange(batch_size, device=sim_matrix.device)

        # Positive similarities: diagonal elements
        pos_sim = sim_matrix[range(batch_size), labels].unsqueeze(1)  # [B, 1]

        # Shift negatives by margin
        margin_matrix = pos_sim - self.margin  # [B, 1]
        logits = sim_matrix - margin_matrix  # broadcast over rows

        # Apply temperature
        logits /= self.temperature

        loss_i2j = F.cross_entropy(logits, labels)
        loss_j2i = F.cross_entropy(logits.T, labels)
        return (loss_i2j + loss_j2i) / 2
    
    def cosine_sim_loss(self, z_i, z_cross):
        z_i_norm = F.normalize(z_i, p=2, dim=1)
        z_cross_norm = F.normalize(z_cross, p=2, dim=1)

        sim = (z_i_norm * z_cross_norm).sum(dim=1)

        loss_sim = (1-sim).mean()
        return loss_sim


    def forward(self, z_iA, z_iB, zt_e, z_o, z_eB_i2g):
        # info_nce = self.info_nce_loss(z_o, z_e)

        info_nce_o = self.info_nce_loss(z_o, z_iB)
        # info_nce_eeB = self.info_nce_loss(z_e, z_be)

        cosine_sim_o = self.cosine_sim_loss(z_iA, z_o)
        cosine_sim_e = self.cosine_sim_loss(zt_e, z_eB_i2g)
        # cosine_sim_be = self.cosine_sim_loss(z_be, z_e)

        cosine = (cosine_sim_o + cosine_sim_e) / 2
        # info_nce = (info_nce_oeB + info_nce_eeB) / 2

        total = info_nce_o + self.alpha*cosine

        return {
            "elem_info_nce": [info_nce_o],
            "elem_cosine_sim": [cosine_sim_o, cosine_sim_e],
            "cosine_sim": cosine,
            "info_nce": info_nce_o,
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
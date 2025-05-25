from Controller.IRESGController.util.misc import NestedTensor

import torch
import torch.nn.functional as F
from torch import nn
from .res_backbone import build_backbone
from .vision_encoder import build_vision_encoder
from .graph_encoder import build_graph_encoder

class CEAtt(nn.Module):
    def __init__(self, vision_encoder, graph_encoder, hidden_dim, nhead, dropout):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.graph_encoder = graph_encoder

        self.attn_vision = nn.MultiheadAttention(hidden_dim, nhead, dropout, batch_first=True)
        self.attn_graph = nn.MultiheadAttention(hidden_dim, nhead, dropout, batch_first=True)
    
    def forward(self, img: NestedTensor, tgt):

        vision, vision_msk, _ = self.vision_encoder(img)
        zt_e, t_mask = self.graph_encoder(tgt)

        # print(vision.size())
        # print(zt_e.size())
        
        z_e, _ = self.attn_graph(
            query=zt_e,
            key=vision,
            value=vision,
            key_padding_mask=vision_msk  # mask cho vision
        )

        print(vision.size())
        print(zt_e.size())
        # print(vision[:, 0].size(),region[:,0].size())

        print(z_e[:,0].size())

        return z_e[:,0]

def build_model(hidden_dim,lr_backbone,masks, backbone, dilation, 
                nhead, nlayer, d_ffn, dropout, random_erasing_prob, activation, pre_train):

    freeze_bert  = True

    vision_backbone = build_backbone(hidden_dim,lr_backbone,masks, backbone, dilation)
    vision_encoder = build_vision_encoder(vision_backbone, hidden_dim, nhead, nlayer, d_ffn, dropout, activation)
    graph_encoder = build_graph_encoder(hidden_dim, nhead, nlayer, d_ffn, dropout, random_erasing_prob, freeze_bert, activation, pre_train)
    
    model = CEAtt(vision_encoder, graph_encoder, hidden_dim, nhead, dropout)

    return model
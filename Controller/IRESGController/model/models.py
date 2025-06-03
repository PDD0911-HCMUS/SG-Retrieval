from Controller.IRESGController.util.misc import NestedTensor

import torch
import torch.nn.functional as F
from torch import nn
from .res_backbone import build_backbone
from .vision_encoder import build_vision_encoder
from .graph_encoder import build_graph_encoder

class CEAtt(nn.Module):
    def __init__(self, vision_encoder, graph_encoder_o, graph_encoder_e, hidden_dim, nhead, dropout):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.graph_encoder_o = graph_encoder_o
        self.graph_encoder_e = graph_encoder_e

        self.attn_graph_o = nn.MultiheadAttention(hidden_dim, nhead, dropout, batch_first=True)
        self.attn_graph_e = nn.MultiheadAttention(hidden_dim, nhead, dropout, batch_first=True)
        self.attn_graph_be = nn.MultiheadAttention(hidden_dim, nhead, dropout, batch_first=True)
    
    def forward(self, img_a: NestedTensor, img_b: NestedTensor, tgt_o, tgt_e):

        z_i, z_i_msk, _ = self.vision_encoder(img_a)
        z_i_b, z_i_b_msk, _ = self.vision_encoder(img_b)

        zt_o, t_mask = self.graph_encoder_o(tgt_o)
        zt_e, t_mask = self.graph_encoder_e(tgt_e)

        # print(vision.size())
        # print(zt_e.size())
        
        z_o, _ = self.attn_graph_o(
            query=zt_o,
            key=z_i,
            value=z_i,
            key_padding_mask=z_i_msk  # mask cho vision
        )

        z_e, _ = self.attn_graph_e(
            query=zt_e,
            key=z_i,
            value=z_i,
            key_padding_mask=z_i_msk  # mask cho vision
        )

        z_be, _ = self.attn_graph_be(
            query=zt_e,
            key=z_i_b,
            value=z_i_b,
            key_padding_mask=z_i_b_msk  # mask cho vision
        )

        print(f"z_i embedding: {z_i[:,0].size()}\nz_o embedding: {z_o[:,0].size()}\nz_e embedding: {z_e[:,0].size()}\nz_be embedding: {z_be[:,0].size()}")

        return  z_i[:,0], z_o[:,0], z_e[:,0], z_be[:,0]
    


def build_model(hidden_dim,lr_backbone,masks, backbone, dilation, 
                nhead, nlayer, d_ffn, dropout, random_erasing_prob, activation, pre_train):

    freeze_bert  = True

    vision_backbone = build_backbone(hidden_dim,lr_backbone,masks, backbone, dilation)
    vision_encoder = build_vision_encoder(vision_backbone, hidden_dim, nhead, nlayer, d_ffn, dropout, activation)
    graph_encoder_o = build_graph_encoder(hidden_dim, nhead, nlayer, d_ffn, dropout, random_erasing_prob, freeze_bert, activation, pre_train)
    graph_encoder_e = build_graph_encoder(hidden_dim, nhead, nlayer, d_ffn, dropout, random_erasing_prob, freeze_bert, activation, pre_train)
    
    model = CEAtt(vision_encoder, graph_encoder_o, graph_encoder_e, hidden_dim, nhead, dropout)

    return model
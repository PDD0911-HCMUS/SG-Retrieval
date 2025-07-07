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

    

    def graph2im(self, z_iA, z_iA_msk, zt_o):
        z_o, _ = self.attn_graph_o(
            query=zt_o,
            key=z_iA,
            value=z_iA,
            key_padding_mask=z_iA_msk  # mask cho vision
        )

        return z_o

    def im2graph(self, z_iA, zt_e, zt_e_mask):

        z_e, _ = self.attn_graph_e(
            query=z_iA,
            key=zt_e,
            value=zt_e,
            key_padding_mask=zt_e_mask  # mask cho triplet
        )

        return z_e
    
    def graph2imB(self, z_iB, z_iB_msk, zt_e):
        z_eB, _ = self.attn_graph_be(
            query=zt_e,
            key=z_iB,
            value=z_iB,
            key_padding_mask=z_iB_msk  # mask cho vision
        )

        return z_eB
    
    def im2graphB(self, z_iB, zt_e, zt_e_mask):
        z_eB, _ = self.attn_graph_be(
            query=z_iB,
            key=zt_e,
            value=zt_e,
            key_padding_mask=zt_e_mask  # mask cho triplet
        )
        return z_eB
    
    def forward(self, img_a: NestedTensor, img_b: NestedTensor, tgt_o, tgt_e):

        z_iA, z_iA_msk, _ = self.vision_encoder(img_a)
        z_iB, z_iB_msk, _ = self.vision_encoder(img_b)

        zt_o, zt_o_mask = self.graph_encoder_o(tgt_o)
        zt_e, zt_e_mask = self.graph_encoder_e(tgt_e)

        # Ask; Graph, Answer: Image -> graph2im
        # trường hợp 
        z_o = self.graph2im(self, z_iA, z_iA_msk, zt_o)
        z_eB_g2i = self.graph2imB(self,  z_iB, z_iB_msk, zt_e)

        # Ask; Image, Answer: Graph -> im2graph
        z_e = self.im2graph(self, z_iA, zt_e, zt_e_mask)
        z_eB_i2g = self.im2graphB(self, z_iB, zt_e, zt_e_mask)

        # Extract cls token from embedding 
        return  z_iA[:,0], z_o[:,0], z_e[:,0], z_eB_g2i[:,0], z_eB_i2g[:,0]
    
def build_model(hidden_dim,lr_backbone,masks, backbone, dilation, 
                nhead, nlayer, d_ffn, dropout, random_erasing_prob, activation, pre_train):

    freeze_bert  = True

    vision_backbone = build_backbone(hidden_dim,lr_backbone,masks, backbone, dilation)
    vision_encoder = build_vision_encoder(vision_backbone, hidden_dim, nhead, nlayer, d_ffn, dropout, activation)
    graph_encoder_o = build_graph_encoder(hidden_dim, nhead, nlayer, d_ffn, dropout, random_erasing_prob, freeze_bert, activation, pre_train)
    graph_encoder_e = build_graph_encoder(hidden_dim, nhead, nlayer, d_ffn, dropout, random_erasing_prob, freeze_bert, activation, pre_train)
    
    model = CEAtt(vision_encoder, graph_encoder_o, graph_encoder_e, hidden_dim, nhead, dropout)

    return model
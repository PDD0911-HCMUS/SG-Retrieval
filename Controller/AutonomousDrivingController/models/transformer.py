# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import torch
from torch import nn

from .driveable_dec import DriveTransformerDecoder, DriveTransformerDecoderLayer
from .entities_dec import EntitiesTransformerDecoder, EntitiesTransformerDecoderLayer
from .hybrid_encoder import HybridTransformerEncoder

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, return_flatten=False):
        super().__init__()

        #HYBRID ENCODER INIT
        self.hybrid_encoder = HybridTransformerEncoder(d_model, nhead, num_encoder_layers,
                                                dim_feedforward, dropout, activation, normalize_before, return_flatten)

        #ENTITIES DECODER INIT
        enitites_decoder_layer = EntitiesTransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        enitites_decoder_norm = nn.LayerNorm(d_model)
        self.enitites_decoder = EntitiesTransformerDecoder(enitites_decoder_layer, num_decoder_layers, enitites_decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        
        # DRIVE DECODER INIT
        drive_decoder_layer = DriveTransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        drive_decoder_norm = nn.LayerNorm(d_model)
        self.drive_decoder = DriveTransformerDecoder(drive_decoder_layer, num_decoder_layers, drive_decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, 
                src, src3, src4, 
                mask, mask3,
                query_embed, query_embed_drive, 
                pos_embed, pos_embed3):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        pos_embed3 = pos_embed3.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        mask3 = mask3.flatten(1)
        
        #====================ENCODER Forward====================#
        memory, sp_memory = self.hybrid_encoder(src, src3, src4, mask, pos_embed)
        #====================End ENCODER Forward====================#

        
        #====================DECODER Forward====================#
        #For Decoder Input
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        query_embed_drive = query_embed_drive.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        tgt_drive = torch.zeros_like(query_embed_drive)
        
        #ENTITIES DECODER Forward 
        hs = self.enitites_decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        
        #DRIVE DECODER Forward -> RETURN Heat Maps 
        hs_sp, hm = self.drive_decoder(tgt_drive, sp_memory, memory_key_padding_mask=mask3,
                          pos=pos_embed3, query_pos=query_embed_drive)
        #====================End DECODER Forward====================#
        
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w), hs_sp.transpose(1, 2), hm


def build_transformer(
        hidden_dim, 
        dropout, 
        nheads, 
        dim_feedforward, 
        enc_layers, 
        dec_layers, 
        pre_norm
    ):
    return Transformer(
        d_model=hidden_dim,
        dropout=dropout,
        nhead=nheads,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate_dec=True,
        return_flatten=True
    )

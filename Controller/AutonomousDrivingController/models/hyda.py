# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .aux_hub import AuxHubHead
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list)

from .backbone import build_backbone
from .transformer import build_transformer
from .driveable_segment import DriveSeg


class HyDA(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_drive_queries, aux_loss=False, training=True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        C_out = [512, 1024]
        self.transformer = transformer
        hidden_dim = transformer.d_model
        nheads = transformer.nhead
        
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_drive = nn.Embedding(num_drive_queries, hidden_dim)
        
        self.input_spatial_proj = nn.Conv2d(C_out[0], hidden_dim, kernel_size=1) #S3
        self.input_hub_proj = nn.Conv2d(C_out[1], hidden_dim, kernel_size=1) #S4
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1) #S5
        
        self.drive_seg = DriveSeg(hidden_dim, nheads)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.backbone = backbone
        self.training = training
        if self.training:
            self.aux_hub_head = AuxHubHead(in_channels=hidden_dim,
                                       num_classes=num_classes + 1,
                                       hidden_dim=hidden_dim)
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See Postfrom models import build_modelProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        #=================BackBone Inference=================#
        features, pos = self.backbone(samples)
            
        src3, mask3 = features[1].decompose() #C3
        src4, mask4 = features[2].decompose() #C4
        src, mask = features[-1].decompose() #C5
        assert mask is not None
        
        src3_proj = self.input_spatial_proj(src3) #S3
        src4_proj = self.input_hub_proj(src4) #S4
        src_proj = self.input_proj(src) #S5
        
        assert mask is not None
        #=================End BackBone Inference=================#
        
        #=================Transformer Inference=================#
        hs, _, hs_sp,  hm = self.transformer(src_proj, src3_proj, src4_proj, 
                                      mask, mask3,
                                      self.query_embed.weight, self.query_embed_drive.weight, 
                                      pos[-1], pos[1])
        #=================End Transformer Inference=================#
        
        #=================Ouput Inference=================#print(f"hs size: {hs.size()}")
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        outputs_seg_masks = self.drive_seg(hs_sp, hm, src3, src3_proj, features)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], "pred_masks": outputs_seg_masks}
        #=================End Ouput Inference=================#
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
            
        if self.training:
            aux_logits, aux_boxes = self.aux_hub_head(src4_proj)
            out["aux_s4_logits"] = aux_logits   # [B, N4, num_classes]
            out["aux_s4_boxes"]  = aux_boxes    # [B, N4, 4]
            
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results

def build_model(
        hidden_dim, 
        position_embedding, 
        lr_backbone, 
        backbone, 
        dilation, 
        return_interm_layers, 
        dropout, 
        nheads, 
        dim_feedforward, 
        enc_layers, 
        dec_layers, 
        pre_norm,
        num_queries,
        num_drive_queries,
        aux_loss,
        num_classes,
        training
    ):

    backbone = build_backbone(hidden_dim, 
                   position_embedding, 
                   lr_backbone, 
                   backbone, 
                   dilation, 
                   return_interm_layers)

    transformer = build_transformer(hidden_dim, 
        dropout, 
        nheads, 
        dim_feedforward, 
        enc_layers, 
        dec_layers, 
        pre_norm)

    model = HyDA(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=num_queries,
        num_drive_queries=num_drive_queries,
        aux_loss=aux_loss,
        training=training
    )
    postprocessors = {'bbox': PostProcess()}

    return model, postprocessors
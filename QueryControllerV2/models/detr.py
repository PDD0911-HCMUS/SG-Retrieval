# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from transformers import GPT2LMHeadModel


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_queries, aux_loss=False):
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
        self.model_llm = GPT2LMHeadModel.from_pretrained('distilgpt2')
        self.projection_layer = nn.Linear(transformer.d_model, self.model_llm.config.n_embd)
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        # Project the transformer output to the GPT-2 input embedding size
        
        projected_hs = self.projection_layer(hs[-1])  # Adjust for GPT-2 input size

        batch_size = projected_hs.size(0)  # Số lượng ảnh trong batch (batch_size)
        num_queries = projected_hs.size(1) 

        projected_hs = projected_hs.unsqueeze(2).expand(-1, -1, 10, -1)  # Thêm chiều seq_len
        projected_hs = projected_hs.view(-1, 10, 768)

        outputs_coord = self.bbox_embed(hs).sigmoid()
        outputs_desc = self.model_llm(inputs_embeds=projected_hs)
        outputs_desc = outputs_desc.logits.view(batch_size, num_queries, 10, -1)
        out = {'pred_boxes': outputs_coord[-1], 'pred_desc': outputs_desc }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_coord, outputs_desc)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord, outputs_desc):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxes': a, 'pred_desc': b}
            for a, b in zip(outputs_coord[:-1], outputs_desc[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, matcher, weight_dict, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

    def loss_text(self, outputs, targets, indices, num_boxes):
        """Compute the loss for the text descriptions (Cross-Entropy Loss for generated text).
        targets dicts must contain the key 'desc_em' (tokenized descriptions) and 'desc_ms' (attention mask).
        """
        assert 'pred_desc' in outputs
        idx = self._get_src_permutation_idx(indices)
        
        # Get predicted text logits
        pred_text_logits = outputs['pred_desc'][idx]  # Shape: (batch_size * num_queries, seq_len, vocab_size)
        
        # Get ground truth tokenized descriptions and masks
        target_text = torch.cat([t['desc_em'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # [batch_size * num_queries, seq_len]
        attention_mask = torch.cat([t['desc_ms'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # [batch_size * num_queries, seq_len]
        
        # Flatten the predicted logits and target_text for CrossEntropyLoss
        pred_text_logits = pred_text_logits.view(-1, pred_text_logits.size(-1))  # [batch_size * num_queries * seq_len, vocab_size]
        target_text = target_text.view(-1)  # [batch_size * num_queries * seq_len]
        attention_mask = attention_mask.view(-1)  # [batch_size * num_queries * seq_len]

        # Compute Cross-Entropy Loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss_text = loss_fct(pred_text_logits, target_text)

        # Mask the loss to exclude padding tokens
        loss_text = loss_text * attention_mask  # Zero out loss where attention mask is 0
        loss_text = loss_text.sum() / num_boxes  # Normalize by number of boxes

        losses = {'loss_text': loss_text}
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error for the number of predicted non-empty boxes."""
        pred_boxes = outputs['pred_boxes']
        device = pred_boxes.device
        tgt_lengths = torch.as_tensor([len(v["boxes"]) for v in targets], device=device)
        card_pred = (pred_boxes[:, :, :].sum(-1) > 0).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'text': self.loss_text
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


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
        out_bbox = outputs['pred_boxes']

        assert len(out_bbox) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        descriptions = outputs.get('pred_desc', None)
        results = [{'boxes': b, 'descriptions': d if descriptions is not None else None} 
                   for b, d in zip(boxes, descriptions)] if descriptions is not None else [{'boxes': b} for b in boxes]

        return results


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


def build(args):
    device = torch.device(args.device)

    # Xây dựng backbone và transformer cho mô hình
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    # Tạo mô hình DETR mà không cần num_classes
    model = DETR(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    
    matcher = build_matcher(args)
    
    weight_dict = {'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_text'] = args.text_loss_coef
    

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    
    losses = ['boxes', 'cardinality', 'text']

    criterion = SetCriterion(matcher=matcher, weight_dict=weight_dict, losses=losses)
    criterion.to(device)
    
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors

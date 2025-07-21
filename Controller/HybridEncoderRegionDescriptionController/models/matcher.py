# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict chứa:
                - "pred_boxes": Tensor [B, N_queries, 4] (cxcywh format)
            targets: list[dict], mỗi dict chứa:
                - "boxes": Tensor [num_gt, 4] (cxcywh format)

        Returns:
            list of size B, mỗi phần tử là tuple:
                (index_pred, index_gt) với len(index_pred) == len(index_gt)
        """
        bs, num_queries = outputs["pred_boxes"].shape[:2]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B * N_queries, 4]
        tgt_bbox = torch.cat([v["boxes"] for v in targets], dim=0)  # [total_gt, 4]

        # Chi phí L1 giữa bounding boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Chi phí GIoU
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Tổng chi phí
        C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        # Tách theo batch
        sizes = [len(v["boxes"]) for v in targets]  # số box trong mỗi ảnh
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64)
            )
            for i, j in indices
        ]


def build_matcher(set_cost_bbox, set_cost_giou):
    return HungarianMatcher(cost_bbox=set_cost_bbox, cost_giou=set_cost_giou)
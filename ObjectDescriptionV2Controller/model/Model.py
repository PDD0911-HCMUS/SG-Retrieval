import torch
import torch.nn.functional as F
from torch import nn
from .GraphEncoder import GraphEncoder
import numpy as np

class G2G(nn.Module):
    def __init__(self, graph_layer_num=6):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.graph_encoder_que = GraphEncoder(d_model=512, nhead=8, nlayer=graph_layer_num,
                                          d_ffn=1024, dropout=0.1, activation="relu")
        
        self.graph_encoder_rev = GraphEncoder(d_model=512, nhead=8, nlayer=graph_layer_num,
                                          d_ffn=1024, dropout=0.1, activation="relu")
        

    def forward(self, graphs_que, graphs_rev):
        output_que, mask_que, pos_que = self.graph_encoder_que(graphs_que)
        output_rev, mask_rev, pos_rev = self.graph_encoder_rev(graphs_rev)

        graph_features_que = output_que[:, 0]
        graph_features_rev = output_rev[:, 0]

        graph_features_que = graph_features_que / graph_features_que.norm(dim=1, keepdim=True)
        graph_features_rev = graph_features_rev / graph_features_rev.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        outputs = {}
        outputs['logits_per_que'] = logit_scale * graph_features_que @ graph_features_rev.t()
        outputs['logits_per_rev'] = outputs['logits_per_que'].t()
        return outputs

class SetCriterion(nn.Module):
    def __init__(self, weight_dict, losses):
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses

    def loss_contrastive(self, outputs, targets, log=True):
        gt = torch.arange(len(outputs['logits_per_que']), device=outputs['logits_per_que'].device).long()
        loss = F.cross_entropy(outputs['logits_per_que'], gt) + F.cross_entropy(outputs['logits_per_rev'], gt)
        losses = {'loss_cont': loss/2}
        return losses


    def get_loss(self, loss, outputs, targets, **kwargs):
        loss_map = {'contrastive': self.loss_contrastive
            }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, **kwargs)

    def forward(self, outputs, targets):
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        return losses

if __name__ == "__main__":
    sample_data_que = {
        'sub_labels': torch.tensor([59,  17, 151, 151, 151, 151, 151, 151, 151, 151]),
        'obj_labels': torch.tensor([101, 106, 151, 151, 151, 151, 151, 151, 151, 151]),
        'rel_labels': torch.tensor([30, 22, 51, 51, 51, 51, 51, 51, 51, 51]),
        'labels': torch.tensor([59,  17, 101, 106, 151, 151, 151, 151, 151, 151])
    }

    sample_data_rev = {
        'sub_labels': torch.tensor([59, 151, 151, 151, 151, 151, 151, 151, 151, 151]),
        'obj_labels': torch.tensor([101, 151, 151, 151, 151, 151, 151, 151, 151, 151]),
        'rel_labels': torch.tensor([30, 51, 51, 51, 51, 51, 51, 51, 51, 51]),
        'labels': torch.tensor([59, 101, 151, 151, 151, 151, 151, 151, 151, 151])
    }

    graphs_que = [sample_data_que]
    graphs_rev = [sample_data_rev]
    # graph_encoder = GraphEncoder(d_model=512, nhead=8, nlayer=6, d_ffn=2048, dropout=0.1, activation="relu")
    model = G2G(graph_layer_num=6)
    output= model(graphs_que, graphs_rev)
    print(output)
    weight_dict = {'loss_cont': 1}
    losses = ['contrastive']

    criterion = SetCriterion(weight_dict=weight_dict, losses=losses)

    loss = criterion(output, None)
    print(loss)

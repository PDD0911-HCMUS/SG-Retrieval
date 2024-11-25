import torch
import torch.nn.functional as F
from torch import nn
from .GraphEncoderV2 import GraphEncoder
import numpy as np

from datasets.dataV3 import build, custom_collate_fn
from torch.utils.data import DataLoader

class G2G(nn.Module):
    def __init__(self, d_model = 256, dropout=0.1, activation="relu", pretrain = 'bert-base-uncased'):
        super().__init__()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.graph_encoder_que = GraphEncoder(d_model=d_model, dropout=dropout, activation=activation, pretrain = pretrain)
        
        self.graph_encoder_rev = GraphEncoder(d_model=d_model, dropout=dropout, activation=activation, pretrain = pretrain)
        

    def forward(self, que, rev):
        out_que = self.graph_encoder_que(que)
        out_rev = self.graph_encoder_rev(rev)

        #mean pooling
        out_que_pooled = out_que.mean(dim=1) 
        out_rev_pooled = out_rev.mean(dim=1)

        #norm L2
        out_que_norm = out_que_pooled / out_que_pooled.norm(dim=1, keepdim=True)
        out_rev_norm = out_rev_pooled / out_rev_pooled.norm(dim=1, keepdim=True)

        print(out_que_norm.size())
        print(out_rev_norm.size())

        logit_scale = self.logit_scale.exp()
        entry = {}
        entry['logits_per_que'] = logit_scale * out_que_norm @ out_rev_norm.t()
        entry['logits_per_rev'] = entry['logits_per_que'].t()

        return entry

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
    
def build_model(d_model = 256, 
                dropout=0.1, 
                activation="relu", 
                pretrain = 'bert-base-uncased', 
                device = 'cpu'
            ):

    weight_dict = {'loss_cont': 1}
    losses = ['contrastive']

    model = G2G(d_model, dropout, activation, pretrain)
    criterion = SetCriterion(weight_dict=weight_dict, losses=losses)

    model.to(device=device)
    criterion.to(device=device)

    return model , criterion

if __name__ == "__main__":
    model = G2G(d_model=256)
    
    device = 'cpu'

    weight_dict = {'loss_cont': 1}
    losses = ['contrastive']

    criterion = SetCriterion(weight_dict=weight_dict, losses=losses)

    ann_file = '/radish/phamd/duypd-proj/SG-Retrieval/Datasets/VisualGenome/Rev.json'
    train_dataset, valid_dataset = build(ann_file)
    dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=custom_collate_fn)

    for que, rev in dataloader:
        que = [{k: v.to(device) for k, v in t.items()} for t in que]
        rev = [{k: v.to(device) for k, v in t.items()} for t in rev]

        entry = model(que, rev)
        print(entry)
        l = criterion(entry, None)
        print(l)
        break

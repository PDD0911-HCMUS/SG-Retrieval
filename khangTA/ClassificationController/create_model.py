import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F

class BERTEmbedding(nn.Module):
    def __init__(self, from_pretrained, num_classes, drop_out):
        super().__init__()
        self.num_classes = num_classes
        self.drop_out = drop_out
        self.bert = BertModel.from_pretrained(from_pretrained)
        self.ffn = nn.Sequential(
            nn.Linear(768, 512),  # 768 -> 512
            nn.ReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(512, 256),  # 512 -> 256
            nn.ReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(256, 128),  # 256 -> 128
            nn.ReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(128, self.num_classes)  # 128 -> num_classes
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        with torch.no_grad():
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            word_embeddings = outputs.last_hidden_state
        cls_token_embedding = word_embeddings[:, 0, :]
        logits = self.ffn(cls_token_embedding) 
        return logits
    
class SetCriterion(nn.Module):
    def __init__(self, num_classes):
        super(SetCriterion, self).__init__()
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets):
        """
        Tính toán loss giữa outputs (logits từ mô hình) và targets (nhãn đúng)
        
        Args:
            outputs (Tensor): Output từ mô hình, kích thước (batch_size, num_classes)
            targets (Tensor): Nhãn thật, kích thước (batch_size,)
        
        Returns:
            loss (Tensor): Giá trị loss giữa outputs và targets.
        """
        # Tính toán loss bằng Cross-Entropy Loss
        loss = self.criterion(outputs, targets)
        return loss
    
def build_model(from_pretrained, num_classes, drop_out):
    model = BERTEmbedding(
        from_pretrained=from_pretrained,
        num_classes=num_classes,
        drop_out=drop_out
    )
    criterion = SetCriterion(num_classes= num_classes)

    return model, criterion

if __name__ == '__main__':
    be = BERTEmbedding(
        from_pretrained='bert-base-uncased',
        num_classes=8,
        drop_out=0.1
    )

    desc_ids = torch.tensor([[  101,  2310, 20098,  2278, 27699,  2072, 24209,  6672,  2102,  1102,
          2239, 12731,  2050,  6187,  2278,  7570,  4907, 13843, 19610,  2654,
          1102, 19098,  3070,  2084,  2232, 15990,  2319,  4229,  1010, 11382,
          2368, 12835,  4048,  1102,  4887, 10722,  2002, 27468, 27793,  4017,
         16371, 10085, 22794,  2100,  1996, 14163,  5063, 27793,  4017, 16371,
         10085,  1010,  6887, 19098,  3070,  2084,  2232, 15990,  2319,  1010,
         24110,  2260,  1012,   102,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0],[  101,  2310, 20098,  2278, 27699,  2072, 24209,  6672,  2102,  1102,
          2239, 12731,  2050,  6187,  2278,  7570,  4907, 13843, 19610,  2654,
          1102, 19098,  3070,  2084,  2232, 15990,  2319,  4229,  1010, 11382,
          2368, 12835,  4048,  1102,  4887, 10722,  2002, 27468, 27793,  4017,
         16371, 10085, 22794,  2100,  1996, 14163,  5063, 27793,  4017, 16371,
         10085,  1010,  6887, 19098,  3070,  2084,  2232, 15990,  2319,  1010,
         24110,  2260,  1012,   102,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0]])
    
    desc_msk = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0]])

    out = be(desc_ids, desc_msk)
    print(out)
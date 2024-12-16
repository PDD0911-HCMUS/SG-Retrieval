import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
import py_vncorenlp
# py_vncorenlp.download_model(save_dir='/home/duypd/ThisPC-DuyPC/SG-Retrieval/khangTA/vncorenlp')

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

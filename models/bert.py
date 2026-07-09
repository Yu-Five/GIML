import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class Bert_Model(nn.Module):
    def __init__(self, args,
                 bert_model_path='./pretrain_weights/bert-base-uncased'):
        super().__init__()

        self.args = args
        self.bert = BertModel.from_pretrained(bert_model_path)
        self.hidden_size = self.bert.config.hidden_size
        # self.fc = nn.Linear(self.hidden_size, 512)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )

        if args.pe:
            self.mu_dul_backbone = nn.Sequential(
                # nn.ReLU(),
                # nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, stride=1, padding=0),
                nn.Linear(512, 512), 
                # nn.BatchNorm2d(self.hidden_size),
                nn.BatchNorm1d(512),
                # nn.ELU(),

            )
            self.logvar_dul_backbone = nn.Sequential(
                # nn.ReLU(),
                # nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, stride=1, padding=0),
                nn.Linear(512, 512), 
                # nn.BatchNorm2d(self.hidden_size),
                nn.BatchNorm1d(512),
                # nn.ELU(),
            )

    def forward(self, encoding):
        # import ipdb; ipdb.set_trace();
        input_ids = encoding['input_ids'].to(self.bert.device)
        attention_mask = encoding['attention_mask'].to(self.bert.device)
        
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        use_pool = False
        # import ipdb; ipdb.set_trace();
        if use_pool: 
            cls_output = bert_outputs.pooler_output
        else: 
            cls_output = bert_outputs.last_hidden_state[:, 0, :]  # [B, hidden_size]
            
        cls_output = self.fc(cls_output)
        
        if self.args.pe: 
            mu = self.mu_dul_backbone(cls_output)          # [B, hidden_size]
            logvar = self.logvar_dul_backbone(cls_output)  # [B, hidden_size]
            std = (0.5 * logvar).exp()             # reparameterization std

            epsilon = torch.randn_like(std)

            if self.training:
                out = mu + epsilon * std
            else:
                out = mu

            return out, mu, std
        else: 
            return cls_output






import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class MetadataFusion(nn.Module):
    def __init__(self, hidden_dim=384, num_fields=4):
        super().__init__()
        config = BertConfig(
            hidden_size=hidden_dim,
            num_attention_heads=6,
            num_hidden_layers=2,
            intermediate_size=hidden_dim * 4,
            max_position_embeddings=num_fields + 1,  # +1 for [CLS]
            num_labels=1
        )
        self.bert = BertModel(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))  # [CLS] token
        
        # Positional embeddings to help identify which field is which
        self.field_type_embed = nn.Embedding(num_fields, hidden_dim)
        self.num_fields = num_fields

    def forward(self, field_embeddings):
        """
        field_embeddings: Tensor of shape [batch_size, num_fields, hidden_dim]
        """
        batch_size = field_embeddings.size(0)
        device = field_embeddings.device
        
        # Add learned field-type embeddings (field identity)
        field_positions = torch.arange(self.num_fields, device=device)
        field_type_bias = self.field_type_embed(field_positions)  # [num_fields, hidden_dim]
        field_type_bias = field_type_bias.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_fields, hidden_dim]
        enriched_fields = field_embeddings + field_type_bias
        
        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # shape: [batch_size, 1, hidden_dim]
        tokens = torch.cat([cls_tokens, enriched_fields], dim=1)  # [batch, num_fields + 1, dim]

        attention_mask = torch.ones(tokens.shape[:2], dtype=torch.long).to(tokens.device)
        output = self.bert(inputs_embeds=tokens, attention_mask=attention_mask)
        fused = output.last_hidden_state[:, 0]  # take [CLS] token
        return fused  # shape: [batch_size, hidden_dim]
    
class MetadataRegressor(nn.Module):
        def __init__(self, hidden_dim=384, num_fields=4):
            super().__init__()
            self.fusion = MetadataFusion(hidden_dim, num_fields)
            self.regressor = nn.Linear(hidden_dim, 1)  # Assuming regression task

        def forward(self, field_embeddings):
            fused = self.fusion(field_embeddings)
            output = self.regressor(fused)
            return output
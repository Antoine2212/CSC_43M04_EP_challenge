import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from torchvision import models
import os
import sys


from topic_model import MetadataFusion
from regression_model import ViewsPredictor

class FullModel(nn.Module):
    def __init__(self, hidden_dim=384, num_fields=3):
        super().__init__()
        self.fusion = MultiModalTransformer(hidden_dim, num_fields)
        self.baseline = ViewsPredictor(num_channels=46, embedding_dim=16, numeric_features=6)
        self.vision = models.resnet50(pretrained=True)
        self.vision.fc = nn.Identity() # Remove the final classification layer
        self.vision_projector = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.topics = MetadataFusion(hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5)) 
        self.regressor = nn.Linear(hidden_dim, 1)  # Assuming regression task

    def forward(self, channel_ids, numeric_features, image_features, topic_embeddings):
        # Get baseline predictions
        baseline_preds = self.baseline(channel_ids, numeric_features)
        # Get vision features
        vision_features = self.vision(image_features)
        vision_features = self.vision_projector(vision_features)
        # Get topic embeddings
        topic_embeddings = self.topics(topic_embeddings)
        # Concatenate all features
        multi_modal_embeddings = torch.cat([baseline_preds.unsqueeze(1), vision_features.unsqueeze(1), topic_embeddings.unsqueeze(1)], dim=1)
        # Pass through transformer
        fused = self.fusion(multi_modal_embeddings)
        output = self.regressor(fused)
        # Add baseline predictions to the output
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        output = alpha * output + (1 - alpha) * baseline_preds
        # Ensure output is squeezed to the correct shape
        output = output.squeeze(1)
        output = torch.exp(output)
        return output

class MultiModalTransformer(nn.Module):
    def __init__(self, hidden_dim=384, num_fields=3):
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

    def forward(self, multi_modal_embeddings):
        """
        multi_modal_embeddings: Tensor of shape [batch_size, num_fields, hidden_dim]
        """
        batch_size = multi_modal_embeddings.size(0)
        device = multi_modal_embeddings.device
        
        # Add learned field-type embeddings (field identity)
        positions = torch.arange(self.num_fields, device=device)
        type_bias = self.field_type_embed(positions)  # [num_fields, hidden_dim]
        type_bias = type_bias.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_fields, hidden_dim]
        enriched_fields = multi_modal_embeddings + type_bias
        
        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # shape: [batch_size, 1, hidden_dim]
        tokens = torch.cat([cls_tokens, enriched_fields], dim=1)  # [batch, num_fields + 1, dim]

        attention_mask = torch.ones(tokens.shape[:2], dtype=torch.long).to(tokens.device)
        output = self.bert(inputs_embeds=tokens, attention_mask=attention_mask)
        fused = output.last_hidden_state[:, 0]  # take [CLS] token
        return fused  # shape: [batch_size, hidden_dim]
        super().__init__()
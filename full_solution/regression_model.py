import torch
import torch.nn as nn

# Define model architecture
class ViewsPredictor(nn.Module):
    def __init__(self, num_channels, embedding_dim=16, numeric_features=6):
        super().__init__()
        
        # Channel embedding
        self.channel_embedding = nn.Embedding(num_channels, embedding_dim)
        
        # Combined features size
        combined_size = embedding_dim + numeric_features
        
        # Regression layers
        self.regressor = nn.Sequential(
            nn.Linear(combined_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, channel_ids, numeric_features):
        # Get channel embeddings
        channel_emb = self.channel_embedding(channel_ids)
        
        # Concatenate embeddings with numeric features
        combined = torch.cat([channel_emb, numeric_features], dim=1)
        
        # Pass through regressor
        return self.regressor(combined).squeeze(1)


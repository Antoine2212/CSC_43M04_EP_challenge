import torch
import torch.nn as nn
import torchvision.models as models

class ResNetFinetune(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True, frozen=False):
        super().__init__()
        # Load the pretrained model
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
        elif backbone == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        # Get feature dimension
        in_features = self.backbone.fc.in_features
        
        # Replace the final classification layer
        self.backbone.fc = nn.Identity()
        
        # Freeze backbone if specified
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Regression head for view prediction
        self.regression_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.ReLU(),  # ReLU for ensuring non-negative view counts
        )

    def forward(self, x):
        x = self.backbone(x["image"])  # Extract features
        x = self.regression_head(x)    # Predict view count
        return x
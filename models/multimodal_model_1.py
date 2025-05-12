import torch
import torch.nn as nn

class MultimodalModel(nn.Module):
    def __init__(self, frozen=False):
        super().__init__()
        
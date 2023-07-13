import torch
from torch import nn
import numpy

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.something = nn.Transformer()
    
    def forward() -> torch.Tensor:
        pass
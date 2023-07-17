import torch
from torch import nn
import numpy


#model data:


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.something = nn.TransformerEncoder()
    
    def forward() -> torch.Tensor:
        pass
import torch.nn as nn

from typing import *

class FeedForwardBlock(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        mlp_ratio: int = 4,
        hidden_features: Optional[int] = None, 
        out_features: Optional[int] = None, 
        drop_rate: float = 0.
    ):
        super().__init__()
        
        if not hidden_features:
            hidden_features = in_features * mlp_ratio
        if not out_features:
            out_features = in_features

        self.linears = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        x = self.linears(x)
        return x
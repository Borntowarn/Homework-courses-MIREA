from typing import *
import torch.nn as nn


class ResidualBlock(nn.Module):
    
    def __init__(self, func: Optional[Callable] = None) -> None:
        super().__init__()
        
        self.func = func
        if not self.func:
            self.func = lambda x: x
    
    def forward(self, x):
        x = self.func(x) + x
        return x
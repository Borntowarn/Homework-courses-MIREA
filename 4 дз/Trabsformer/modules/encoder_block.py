import torch.nn as nn

from .mha_block import MHABlock
from .residual_block import ResidualBlock
from .feed_forward_block import FeedForwardBlock


class EncoderBlock(nn.Module):
    def __init__(
        self, 
        emb_len: int, 
        num_heads: int = 8, 
        mlp_ratio: int = 4, 
        drop_rate: float = 0.
    ) -> None:
        super().__init__()

        self.first_residual = ResidualBlock(
            nn.Sequential(
                nn.LayerNorm(emb_len),
                MHABlock(emb_len, num_heads, drop_rate, drop_rate),
                nn.Dropout(drop_rate)
            )
        )
        
        self.second_residual = ResidualBlock(
            nn.Sequential(
                nn.LayerNorm(emb_len),
                FeedForwardBlock(emb_len, mlp_ratio),
                nn.Dropout(drop_rate)
            )
        )           

    def forward(self, x):
        
        x = self.first_residual(x)
        x = self.second_residual(x)
        
        return x
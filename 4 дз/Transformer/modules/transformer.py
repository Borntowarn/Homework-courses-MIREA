import torch.nn as nn

from .encoder_block import EncoderBlock


class Transformer(nn.Module):
    def __init__(
        self, 
        num_layers: int, 
        emb_len: int, 
        num_heads: int = 12,
        mlp_ratio: int = 4,
        drop_rate: float = 0.
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(emb_len, num_heads, mlp_ratio, drop_rate)
            for i in range(num_layers)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
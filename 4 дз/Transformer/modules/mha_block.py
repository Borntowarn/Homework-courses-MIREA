import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class MHABlock(nn.Module):
    def __init__(
        self,
        emb_len: int,
        num_heads: int = 8,
        attn_drop: float = 0.,
        out_drop: float = 0.
    ) -> None:
        super().__init__()
        
        self.num_heads = num_heads # number of heads
        head_emb = emb_len // num_heads # embeddings length after head
        self.scale = head_emb ** -0.5 # scale param for decrease dispersion

        self.qkv = nn.Linear(emb_len, emb_len * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.out = nn.Sequential(
            nn.Linear(emb_len, emb_len),
            nn.Dropout(out_drop)
        )
        

    def forward(self, x):
        
        QKV = self.qkv(x)
        """
        b - batch
        l - sequence length (number of patches)
        n - 3 (Q K V)
        h - num heads
        hl - seq length after attention
        """
        Q, K, V = rearrange(QKV, 'b l (n h hl) -> n b h l hl', n = 3, h = self.num_heads)

        attention = F.softmax(torch.einsum('bhqo, bhko -> bhqk', Q, K) / self.scale, dim=-1)
        attention = self.attn_drop(attention)
        attention = attention @ V
        attention = rearrange(attention, 'b h l hl -> b l (h hl)')
        
        out = self.out(attention)
        return out

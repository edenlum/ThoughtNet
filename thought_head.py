import torch
import torch.nn as nn
from copy import copy
from .model import MultiHeadedSelfAttention

# Thought Head takes the output sequence of the transformer and returns another sequence of length input sequence + 1
class ThoughtHead(nn.Module):
    def __init__(self, dim, d_ff, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = dim
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, dim),
            nn.Dropout(dropout)
        )
        self.attn = MultiHeadedSelfAttention(dim, num_heads, dropout=dropout)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        
    def forward(self, input):
        # x is b, s, d
        x = input
        b, s, d = x.shape
        # b, s, d -> b, s+1, d
        x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
        x = self.attn(x, None)
        x = self.ff(self.norm(x))
        x = torch.cat((x[:, 0:1], input[:, 1:]))
        return x
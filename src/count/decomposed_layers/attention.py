# Torch & Numpy
import torch
import torch.nn as nn
import numpy as np

# Local
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

# Local
from src.count.decomposed_layers.linear import KLinear


class KMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()

        self.d_model = embed_dim
        self.d_k = embed_dim // num_heads
        self.h = num_heads

        self.q_linear = KLinear(embed_dim, embed_dim)
        self.v_linear = KLinear(embed_dim, embed_dim)
        self.k_linear = KLinear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None):

        bs = query.size(0)

        # perform linear operation and split into h heads

        key = self.k_linear(key).view(bs, -1, self.h, self.d_k)
        query = self.q_linear(query).view(bs, -1, self.h, self.d_k)
        value = self.v_linear(value).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)  # calculate attention using function we will define next
        scores = self.attention(query, key, value, self.d_k, attn_mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output

    def attention(self, q, k, v, d_k, attn_mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
        if attn_mask is not None:
            L, S = attn_mask.shape
            attn_mask = attn_mask.view(1, 1, L, S)  # L, 1, S
            scores = scores.masked_fill(attn_mask.bool(), -1e9)
        scores = torch.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)
        return output

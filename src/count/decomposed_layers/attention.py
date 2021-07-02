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


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        "Standard scaled dot-product attention"
        super().__init__()

    def forward(self, query, key, value, attn_mask, key_padding_mask):
        """Standard scaled dot-product attention

        Arguments
        ---------
        query : torch.Tensor
            Tensor of shape [B, L, D]
        key : torch.Tensor
            Tensor of shape [B, S, D]
        value : torch.Tensor
            Tensor of shape [B, S, D]
        key_padding_mask : torch.Tensor
            Tensor of shape [B, S]
        attn_mask : torch.Tensor
            Tensor of shape [L, S]

        Returns
        -------

        """
        B, L, D = query.shape
        scores = torch.bmm(query, key.transpose(2, 1))  # B, L, S
        scores = scores / np.sqrt(D)  # scale by sqrt(d)

        # fill the masked values with super small number
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1)  # B, L, S
            scores = scores.masked_fill(attn_mask.bool(), -1e9)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).expand(-1, L, -1)  # B, L, S
            scores = scores.masked_fill(key_padding_mask.bool(), -1e9)

        attn = torch.softmax(scores, dim=-1)  # B, L, S
        context = torch.bmm(attn, value)  # shape: B, L, D
        return context, attn


class KMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # Transformations
        self.W_q = KLinear(self.embed_dim, self.num_heads * self.embed_dim)
        self.W_k = KLinear(self.embed_dim, self.num_heads * self.embed_dim)
        self.W_v = KLinear(self.embed_dim, self.num_heads * self.embed_dim)

        # Output
        self.linear = KLinear(self.num_heads * self.embed_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor,
        attn_mask: torch.Tensor,
    ):
        # query shape: batch, output_size, embed_dim
        # key shape: batch, input_size, embed_dim
        # value shape: batch, input_size, embed_dim
        B, L, D = query.shape
        residual = query

        # Construction of query/key/value
        # ===============================
        # shapes: B, H, L/S, D
        # where H = num_heads and L/S = input/output dims
        q_s = self.W_q(query).view(B, -1, self.num_heads, self.embed_dim).transpose(1, 2)
        k_s = self.W_k(key).view(B, -1, self.num_heads, self.embed_dim).transpose(1, 2)
        v_s = self.W_v(value).view(B, -1, self.num_heads, self.embed_dim).transpose(1, 2)

        if attn_mask is not None:
            # shape: [B, H, L, S]
            attn_mask = attn_mask.unsqueeze(0).repeat(B, self.num_heads, 1, 1).bool()
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(B, 1, 1, -1).repeat(1, self.num_heads, L, 1)
            key_padding_mask.bool()

        # Scaled dot-product attention
        # ============================
        scores = torch.matmul(q_s, k_s.transpose(-1, -2))
        scores = scores / np.sqrt(self.embed_dim)  # scale by sqrt(d)

        # fill the masked values with super small number
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -1e9)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask, -1e9)

        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v_s)  # shape: B, H, L, D

        context = context.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.embed_dim)

        output = self.linear(context)
        # return output and weights
        return output, attn

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        millions = total // 1_000_000
        thousands = (total - millions * 1_000_000) // 1_000
        string = str(millions) + "." + str(thousands) + "M"
        return string

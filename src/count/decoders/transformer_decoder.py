# STL
import sys
from pathlib import Path
from typing import Optional

# Torch
import torch.nn as nn
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Local
from src.count.decoders.base_decoder import Decoder


@Decoder.register("gpt2-transformer-decoder")
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_attention_heads: int,
        num_layers: int,
        dropout: float,
        **kwargs,
    ) -> None:
        """Simple Transformer-Decoder model (no encoder at all).

        Arguments
        ---------
        input_dim : int
            Embedding dimension of inputs.
        hidden_dim : int
            Dimension to use for decoded vectors
        num_attention_heads : int
            Number of attention heads to use.
        num_layers : int
            Number of decoder blocks to use.
        dropout : float
            Float between 0.0 and 1.0, probability of dropout.
        activation : Optional[str]
            Default is "relu"
        """
        super().__init__(**kwargs)
        decoder_layers = []
        for i in range(num_layers):
            layer = TransformerDecoderLayer(
                input_dim=input_dim,
                num_attention_heads=num_attention_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            decoder_layers.append(layer)

        self.decoder_layers = nn.ModuleList(decoder_layers)

    def forward(
        self,
        target: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        for layer in self.decoder_layers:
            print(target)
            target = layer(
                target=target,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )

        return target


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_attention_heads: int,
        hidden_dim: int,
        dropout: float,
        **kwargs,
    ) -> None:
        """Simple Transformer-Decoder block (no encoder at all).

        Arguments
        ---------
        input_dim : int
            Embedding dimension of inputs.
        num_attention_heads : int
            Number of attention heads to use.
        hidden_dim : int
            Dimension to use for decoded vectors
        dropout : float
            Float between 0.0 and 1.0, probability of dropout.
        """
        super().__init__(**kwargs)

        # attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        # FF network
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # dropout
        self.dropout = nn.Dropout(dropout)

        # norms
        self.layer_norm_1 = nn.LayerNorm(input_dim, eps=1e-12)
        self.layer_norm_2 = nn.LayerNorm(input_dim, eps=1e-12)

    def forward(
        self,
        target: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attention + feedfoward network

        Arguments
        ---------
        target : torch.Tensor
            Sequence of embeddings to decode, of shape `(L, B, D)`
        attn_mask : torch.Tensor
            Binary matrix indicating which items in `target` to attend to (1) or ignore (0) at each
            timestep. Shape is `(L, L)`
        key_padding_mask : torch.Tensor
            Binary matrix indicating which items in `target` are padding.

        Returns
        -------
        torch.Tensor :
            Decoded tensor of shape `(N, B, embedding_dim)`
        """
        # prenorm
        target = self.layer_norm_1(target)

        # attention & dropout
        inp, _ = self.attention(
            target,
            target,
            target,
            attn_mask=attn_mask,
            need_weights=False,
        )
        inp = self.dropout(inp)

        # add
        target = inp + target

        # prenorm
        target = self.layer_norm_2(target)

        # feedfoward & dropout
        inp = self.feedforward(target)
        inp = self.dropout(inp)

        # add
        target = inp + target
        return target


if __name__ == "__main__":
    # just a test
    decoder = TransformerDecoder(128, 2, 4, 64, 0.2, "relu", None)
    trg = torch.ones(4, 30, 128)
    mask = torch.tril(torch.ones(30, 30))
    mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
    print(decoder(trg, mask))

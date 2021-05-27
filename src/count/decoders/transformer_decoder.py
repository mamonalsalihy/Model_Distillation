# STL
import sys
from pathlib import Path
from typing import List, Optional

# Torch
import torch.nn as nn
import torch

# AllenNLP
from allennlp.models import Model
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.layer_norm import LayerNorm
from allennlp.nn.activations import Activation

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
        activation: Optional[str] = "relu",
        norm: Optional[LayerNorm] = None,
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
        self.norm = norm
        decoder_layers = []
        for i in range(num_layers):
            layer = TransformerDecoderLayer(
                input_dim=input_dim,
                num_attention_heads=num_attention_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
                activation=activation,
            )
            decoder_layers.append(layer)

        self.decoder_layers = nn.ModuleList(decoder_layers)

    def forward(
        self,
        target: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        for layer in self.decoder_layers:
            target = layer(target, attn_mask, key_padding_mask)

        if self.norm is not None:
            target = self.norm(target)
        return target


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_attention_heads: int,
        hidden_dim: int,
        dropout: float,
        activation: Optional[str] = "relu",
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
        activation : Optional[str]
            Default is "relu"
        """
        super().__init__(**kwargs)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_attention_heads, dropout=dropout
        )

        self.feedforward = FeedForward(
            input_dim=input_dim,
            num_layers=2,
            hidden_dims=[hidden_dim, input_dim],
            activations=Activation.by_name(activation)(),
            dropout=dropout,
        )

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
            Sequence of embeddings to decode, of shape `(batch_size, N, embedding_dim)`
        attn_mask : torch.Tensor
            Binary matrix indicating which items in `target` to attend to (1) or ignore (0) at each
            timestep. Shape is `(batch_size, N, N)`
        key_padding_mask : torch.Tensor
            Binary matrix indicating which items in `target` are padding.

        Returns
        -------
        torch.Tensor :
            Decoded tensor of shape `(batch_size, N, embedding_dim)`
        """
        target = target.permute(1, 0, 2)
        attn_target, weights = self.self_attn(
            key=target,
            value=target,
            query=target,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        # not sure why we add instead of just use attn_target but that's how they do it in pytorch
        target = target + attn_target
        return self.feedforward(target.permute(1, 0, 2))


if __name__ == "__main__":
    # just a test
    decoder = TransformerDecoder(128, 2, 4, 64, 0.2, "relu", None)
    trg = torch.ones(4, 30, 128)
    mask = torch.tril(torch.ones(30, 30))
    mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
    print(decoder(trg, mask))

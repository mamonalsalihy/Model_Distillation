# STL
import sys
from pathlib import Path
from typing import Optional

# Torch
import torch.nn as nn
import torch

# Modules
from allennlp.modules.augmented_lstm import AugmentedLstm

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Local
from src.count.decoders.base_decoder import Decoder


@Decoder.register("lstm-decoder")
class LSTMDecoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_layers: int,
            use_highway=False,
            go_forward=True,
    ) -> None:
        """Simple LSTM-Decoder model (no encoder at all).

        Arguments
        ---------
        input_dim : int
            Embedding dimension of inputs.
        hidden_dim : int
            Dimension to use for decoded vectors
        num_layers : int
            Number of decoder blocks to use.
        """
        super().__init__()
        decoder_layers = []
        for i in range(num_layers):
            layer = AugmentedLstm(input_size=input_dim,
                                  hidden_size=hidden_dim,
                                  use_highway=use_highway,
                                  go_forward=go_forward
                                  )
            decoder_layers.append(layer)

        self.decoder_layers = nn.ModuleList(decoder_layers)

    def forward(
            self,
            packed_input: torch.Tensor,
    ):
        for layer in self.decoder_layers:
            decoded, hidden = layer(
                packed_input,
            )

        return decoded, hidden

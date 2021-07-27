# STL
import sys
from pathlib import Path
from typing import Optional
from typing import Optional, Tuple, List

# Torch
import torch.nn as nn
import torch

# Modules
from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.common.checks import ConfigurationError

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Local
from src.count.decoders.base_decoder import Decoder

TensorPair = Tuple[torch.Tensor, torch.Tensor]


@Decoder.register("lstm-decoder")
class LSTMDecoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_layers: int,
            use_highway=False,
            go_forward=True,
            recurrent_dropout_probability=0.0,
            layer_dropout_probability=0.0
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
        lstm_input_size = input_dim
        for i in range(num_layers):
            layer = AugmentedLstm(input_size=lstm_input_size,
                                  hidden_size=hidden_dim,
                                  use_highway=use_highway,
                                  go_forward=go_forward,
                                  recurrent_dropout_probability=recurrent_dropout_probability
                                  )
            self.add_module('layer_{}'.format(i), layer)
            decoder_layers.append(layer)
            lstm_input_size = hidden_dim

        self.decoder_layers = decoder_layers
        self.layer_dropout = InputVariationalDropout(layer_dropout_probability)

    def forward(
            self,
            packed_input: torch.Tensor,
            initial_state: Optional[TensorPair] = None
    ):
        if initial_state is None:
            hidden_states: List[Optional[TensorPair]] = [None] * len(self.decoder_layers)
        elif initial_state[0].size()[0] != len(self.decoder_layers):
            raise ConfigurationError(
                "Initial states were passed to forward() but the number of "
                "initial states does not match the number of layers."
            )
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0), initial_state[1].split(1, 0)))

        packed_output = packed_input
        final_h = []
        final_c = []
        for i, state in enumerate(hidden_states):
            layer = getattr(self, "layer_{}".format(i))
            packed_output, final_states = layer(packed_output, state)

            if 1 < len(self.decoder_layers) - 1:
                packed_output = self.layer_dropout(packed_output)
            final_h.extend(final_states[0])
            final_c.extend(final_states[1])

        final_h = torch.cat(final_h, dim=0)
        final_c = torch.cat(final_c, dim=0)

        final_state_tuple = (final_h, final_c)
        return packed_output, final_state_tuple

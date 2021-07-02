# STL
import logging
import sys
import copy
from pathlib import Path
from typing import Dict

# Torch transformer
import torch
import torch.nn as nn

# AllenNLP
from allennlp.data import Vocabulary

# Models
from allennlp.models import Model
from allennlp.modules import Embedding
from allennlp.data import TensorDict

sys.path.append(str(Path(__file__).resolve().parents[3]))

# Local
from src.count.decoders.base_decoder import Decoder
from src.count.models.base_transformer import Transformer

logger = logging.getLogger(__name__)


@Model.register("bidirectional-language-model", exist_ok=True)
class BidirectionalTransformer(Transformer):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: Embedding,
        pos_embedder: Embedding,
        decoder: Decoder,
        embedding_dim: int,
    ) -> None:
        super().__init__(vocab, embedder, decoder, embedding_dim)
        self.backward = False

    def _forward_helper(self, tokens: TensorDict):
        pass

    def forward(
        self,
        tokens: TensorDict,
    ) -> Dict[str, torch.Tensor]:
        token_ids = tokens["tokens"]["tokens"]

        self.backward = False
        forward = super().forward(tokens)

        self.backward = True
        tokens["tokens"]["tokens"] = torch.fliplr(token_ids)
        backward = super().forward(tokens)

        forward_logits = forward["logits"]  # Logits for tokens 2 -> N
        backward_logits = backward["logits"]  # Logits for tokens (N-1) -> 1

        # Since each direction misses a single prediction (i.e., BOS and EOS), we have to make a new
        # tensor with the right number of tokens.
        B, Nm1, D = forward_logits.shape
        logits = torch.zeros(size=(B, Nm1 + 1, D), device=forward_logits.device, dtype=torch.float)

        logits[:, 1:, :] += forward_logits  # 2 -> N
        logits[:, :-1, :] += backward_logits  # 1 -> N-1

        # return combined logits & loss
        return {
            "logits": logits,
            "loss": forward["loss"] + backward["loss"],
        }

# STL
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

# Torch
import torch
import torch.nn as nn

# AllenNLP
from allennlp.data import Vocabulary
from allennlp.data import TensorDict

# Models
from allennlp.models import Model
from allennlp.modules import Embedding

sys.path.append(str(Path(__file__).resolve().parents[3]))

# Local
from src.count.decoders.base_decoder import Decoder
from src.count.models.base_transformer import Transformer

logger = logging.getLogger(__name__)


@Model.register("simple-transformer-language-model", exist_ok=True)
class SimpleTransformerLanguageModel(Transformer):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: Embedding,
        pos_embedder: Embedding,
        decoder: Decoder,
        embedding_dim: int,
        state_dict: Optional[str] = None,
        backward: Optional[bool] = False,
    ) -> None:
        super().__init__(vocab, embedder, pos_embedder, decoder, embedding_dim, state_dict)

        self.backward = backward

    def _add_positional_embeddings(self, emb):
        # emb: [S, B, D]
        positions = torch.arange(len(emb), device=emb.device).unsqueeze(-1)
        # If we're going backwards, flip the positions around since the tokens are also backwards.
        if self.backward:
            positions = torch.flip(positions, dims=[0])
        emb = emb + self.pos_embedder(positions).expand_as(emb)
        return emb

    def forward(
        self,
        tokens: torch.Tensor,
        ratio: float,
        only_predict_next: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Flip them around if it's backwards
        if self.backward:
            tokens = torch.flip(tokens, dims=[1])  # shape: [B, S]
        return super().forward(tokens, ratio, only_predict_next)

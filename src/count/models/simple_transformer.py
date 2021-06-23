# STL
import logging
import sys
import numpy
from pathlib import Path
from typing import Dict

# Torch
import torch
import torch.nn as nn

# AllenNLP
from allennlp.data import Vocabulary
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.common.registrable import Registrable

# Models
from allennlp.models import Model
from allennlp.modules import Embedding, TextFieldEmbedder

# Layers
from allennlp.nn.util import get_text_field_mask

# Inference
from allennlp.predictors.predictor import Predictor

# Training
from allennlp.training.metrics import Perplexity

sys.path.append(str(Path(__file__).resolve().parents[3]))

# Local
from src.count import config
from src.count.data import WikiTextReader
from src.count.decoders.base_decoder import Decoder
from src.count.models.base_transformer import Transformer

logger = logging.getLogger(__name__)


@Model.register("simple-transformer-language-model", exist_ok=True)
class SimpleTransformerLanguageModel(Transformer):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        decoder: Decoder,
        embedding_dim: int,
        max_positions: int,
        backward: bool = False,
    ) -> None:
        super().__init__(vocab, embedder, decoder, embedding_dim)

        self.pos_embedder = nn.Embedding(max_positions, embedding_dim)
        self.backward = backward

    def _add_positional_embeddings(self, token_ids, embeddings):
        positions = torch.arange(token_ids.shape[1], device=embeddings.device).unsqueeze(-1)
        pos_embeddings = self.pos_embedder(positions).permute(1, 0, 2).expand_as(embeddings)
        return embeddings + pos_embeddings

    def forward(
        self,
        tokens: TextFieldTensors,
    ) -> Dict[str, torch.Tensor]:
        token_ids = tokens["tokens"]["tokens"]

        # Flip them around if it's backwards
        if self.backward:
            tokens["tokens"]["tokens"] = torch.fliplr(token_ids)
        return super().forward(tokens)

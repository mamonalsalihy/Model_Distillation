# STL
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy
import torch

# Torch transformer
import torch.nn as nn

# AllenNLP
from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer

# Models
from allennlp.models import Model
from allennlp.modules import Embedding, TextFieldEmbedder

# Layers
from allennlp.modules.attention import Attention
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.transformer import TransformerLayer, TransformerStack
from allennlp.modules.transformer.positional_encoding import SinusoidalPositionalEncoding
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.initializers import InitializerApplicator

# Inference
from allennlp.predictors.predictor import Predictor

# Training
from allennlp.training.metrics import Perplexity
from allennlp.training.trainer import GradientDescentTrainer, Trainer

sys.path.append(str(Path(__file__).resolve().parents[3]))

# Local
from src.count import config
from src.count.data import WikiTextReader
from src.count.decoders.base_decoder import Decoder
from src.count.models.base_transformer import Transformer

logger = logging.getLogger(__name__)


@Model.register("bidirection-transformer-language-model", exist_ok=True)
class BiDirectionTransformerLanguageModel(Transformer):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        decoder: Decoder,
        embedding_dim: int,
        max_positions: int,
    ) -> None:
        super().__init__(vocab, embedder, decoder, embedding_dim)
        self.pos_embedder = nn.Embedding(max_positions, embedding_dim)

    def _add_positional_embeddings(self, token_ids, embeddings):
        positions = torch.arange(token_ids.shape[1], device=embeddings.device).unsqueeze(-1)
        pos_embeddings = self.pos_embedder(positions).permute(1, 0, 2).expand_as(embeddings)
        return embeddings + pos_embeddings

    def _make_attention_mask(self, target_len, context_len):
        mask_values = torch.full(
            (target_len - 1,),
            fill_value=-float("inf"),
        )
        attn_mask = torch.diag(mask_values, diagonal=1)
        return attn_mask

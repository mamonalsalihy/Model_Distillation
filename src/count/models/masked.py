# STL
import logging
import sys
from pathlib import Path

# Torch transformer
import torch
import torch.nn as nn

# AllenNLP
from allennlp.data import Vocabulary

# Models
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder

sys.path.append(str(Path(__file__).resolve().parents[3]))

# Local
from src.count.decoders.base_decoder import Decoder
from src.count.models.base_transformer import Transformer

logger = logging.getLogger(__name__)


@Model.register("masked-language-model", exist_ok=True)
class MaskedLanguageModelTransformer(Transformer):
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

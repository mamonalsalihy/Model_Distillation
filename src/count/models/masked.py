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
from allennlp.modules import Embedding

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
        embedder: Embedding,
        pos_embedder: Embedding,
        decoder: Decoder,
        embedding_dim: int,
        state_dict: Optional[str] = None,
    ) -> None:
        super().__init__(vocab, embedder, pos_embedder, decoder, embedding_dim, state_dict)

    def _make_attention_mask(self, emb):
        size = emb.size(0)
        mask_values = torch.full(
            (size - 1,),
            fill_value=-float("inf"),
        )
        attn_mask = torch.diag(mask_values, diagonal=1)
        return attn_mask

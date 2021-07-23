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


@Model.register("student-language-model", exist_ok=True)
class StudentLanguageModel(Transformer):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: Embedding,
        pos_embedder: Embedding,
        decoder: Decoder,
        embedding_dim: int,
        teacher: Model,
        temperature: float,
        hard_label_weight: float,
        state_dict: Optional[str] = None,
    ) -> None:
        super().__init__(vocab, embedder, pos_embedder, decoder, embedding_dim, state_dict)

        self.teacher = teacher
        teacher.eval()
        self._initialize_embeddings()

        self.alpha = hard_label_weight
        self.T = temperature

        # Distillation
        self.kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    def _add_positional_embeddings(self, emb):
        # emb: [S, B, D]
        positions = torch.arange(len(emb), device=emb.device).unsqueeze(-1)
        emb = emb + self.pos_embedder(positions).expand_as(emb)
        return emb

    def _initialize_embeddings(self):
        self.embedder.weight.data = self.teacher.embedder.weight.data.clone()
        self.pos_embedder.weight.data = self.teacher.pos_embedder.weight.data.clone()

    def kl_loss(self, logits, softs):
        s = torch.log_softmax(logits / self.T, dim=-1)
        t = torch.softmax(softs / self.T, dim=-1)

        return (self.T ** 2) * self.kl_loss_fn(s, t)

    def forward(
        self,
        tokens: torch.Tensor,
        ratio: float,
        only_predict_next: bool = False,
    ) -> Dict[str, torch.Tensor]:

        # Get teacher outputs first, before we flip the tokens to [S, B]
        with torch.no_grad():
            t_out = self.teacher(tokens, ratio)
            t_logits = t_out["logits"]
            t_loss = t_out["loss"]

        tokens = tokens.transpose(0, 1)  # new shape [S+1, B]
        source = tokens[:-1]  # [S, B]
        labels = tokens[1:]  # [S, B]

        # Get embeddings
        # ==============
        embeddings = self.embedder(source)  # shape: [S, B, D]
        embeddings = self._add_positional_embeddings(embeddings)

        # Make prediction
        # ===============
        logits = self._predict(embeddings)

        if only_predict_next:  # inference; only care about final value
            logits = logits[-1:]  # shapes: [1, B]
            labels = labels[-1:]
            t_logits = t_logits[-1:]

        # Calculate loss
        # ==============
        preds = logits.reshape(-1, self.vocab_size)
        reals = labels.reshape(-1)
        softs = t_logits.reshape(-1, self.vocab_size)
        kl_loss = self.kl_loss(preds, softs)
        ce_loss = self.loss(preds, reals)

        loss = (1 - self.alpha) * kl_loss + self.alpha * ce_loss

        self.perplexity(ce_loss)
        self.word_perplexity(ce_loss * ratio)
        return {"logits": logits, "loss": loss}

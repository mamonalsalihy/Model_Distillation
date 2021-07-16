# STL
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy

# Torch
import torch
import torch.nn as nn

# AllenNLP
from allennlp.data import Vocabulary

# Models
from allennlp.models import Model
from allennlp.modules import Embedding

# Layers
from allennlp.nn.util import get_text_field_mask

# Training
from allennlp.training.metrics import Perplexity

sys.path.append(str(Path(__file__).resolve().parents[3]))

# Local
from src.count import config
from src.count.decoders.base_decoder import Decoder

logger = logging.getLogger(__name__)


@Model.register("base-transformer", exist_ok=True)
class Transformer(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: Embedding,
        pos_embedder: Embedding,
        decoder: Decoder,
        embedding_dim: int,
        state_dict: Optional[str] = None,
    ) -> None:
        super().__init__(vocab)

        # Vocabulary stuff
        # ================
        self.vocab_size = vocab.get_vocab_size()
        self.PAD_IDX = self.vocab.get_token_index(config.PAD)

        # TransformerDecoder stuff
        # ========================
        self.embedder = embedder
        self.pos_embedder = pos_embedder
        self.decoder = decoder

        # Language modeling head
        # ======================
        # linear layer that maps the last last transformer layer to logits for each word
        self.lm_head = torch.nn.Linear(embedding_dim, self.vocab_size, bias=False)
        self.lm_head.weight = self.embedder.weight

        # Evaluation
        # ==========
        self.perplexity = Perplexity()
        self.word_perplexity = Perplexity()
        self.loss = nn.CrossEntropyLoss(ignore_index=self.PAD_IDX, reduction="mean")

        # Initialize weights
        # ==================
        logger.info("Number of parameters: %s", self.count_parameters())
        logger.info("Initializing...")
        self.apply(self.init_weights)

        # load weights if necessary
        if state_dict:
            logger.info(f"Loading pretrained weights from {state_dict}...")
            self.load_state_dict(torch.load(state_dict))

    def _add_positional_embeddings(self, emb):
        positions = torch.arange(len(emb), device=emb.device).unsqueeze(-1)
        emb = emb + self.pos_embedder(positions).expand_as(emb)
        return emb

    def _make_attention_mask(self, emb: torch.Tensor) -> torch.Tensor:
        size = emb.size(0)
        attn_mask = torch.full(
            (size, size),
            -float("Inf"),
            device=emb.device,
            dtype=emb.dtype,
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)
        return attn_mask

    def _predict(self, target: torch.Tensor):
        # target: [S, B, D]
        attn_mask = self._make_attention_mask(target)
        decoded = self.decoder(target=target, attn_mask=attn_mask)
        logits = self.lm_head(decoded)  # shape (batch_size, seq_len, vocab_size)
        return logits

    def forward(
        self,
        tokens: torch.Tensor,
        ratio: float,
        only_predict_next: bool = False,
    ) -> Dict[str, torch.Tensor]:
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

        # Calculate loss
        # ==============
        preds = logits.reshape(-1, self.vocab_size)
        reals = labels.reshape(-1)
        loss = self.loss(preds, reals)

        self.perplexity(loss)
        self.word_perplexity(loss * ratio)
        return {"logits": logits, "loss": loss}

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Takes logits from `forward` and computes the corresponding label

        Arguments
        ---------
        output_dict : Dict[str, torch.Tensor]
            Dictionary returned by `forward`. Must contain a key with `logits`.
        Returns
        -------
        Dict[str, torch.Tensor]:
            Same as input dictionary, but with another key `label` indicating the predicted label
        """
        # Take the logits from the forward pass, and compute the label IDs for maximum values
        logits = output_dict["logits"].cpu().data.numpy()
        predicted_id = numpy.argmax(logits, axis=-1)

        # Convert these IDs back to label strings using vocab
        output_dict["tokens"] = [
            self.vocab.get_token_from_index(x, namespace="tokens") for x in predicted_id.ravel()
        ]
        output_dict["token_ids"] = list(predicted_id.ravel())
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "perplexity": self.perplexity.get_metric(reset),
            "word_perplexity": self.word_perplexity.get_metric(reset),
        }

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        millions = total // 1_000_000
        thousands = (total - millions * 1_000_000) // 1_000
        string = str(millions) + "." + str(thousands) + "M"
        return string

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

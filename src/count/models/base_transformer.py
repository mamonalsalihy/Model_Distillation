# STL
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy

# Torch
import torch
import torch.nn as nn

# AllenNLP
from allennlp.data import Vocabulary
from allennlp.data.fields.text_field import TextFieldTensors

# Models
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder

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
        embedder: TextFieldEmbedder,
        decoder: Decoder,
        embedding_dim: int,
    ) -> None:
        super().__init__(vocab)

        # Vocabulary stuff
        # ================
        self.vocab_size = vocab.get_vocab_size()
        self.PAD_IDX = self.vocab.get_token_index(config.PAD)

        # TransformerDecoder stuff
        # ========================
        self.embedder = embedder
        self.decoder = decoder

        # Language modeling head
        # ======================
        # linear layer that maps the last last transformer layer to logits for each word
        self.lm_head = torch.nn.Linear(embedding_dim, self.vocab_size, bias=False)
        self.lm_head.weight = self.embedder._token_embedders["tokens"].weight

        # Evaluation
        # ==========
        self.metric = Perplexity()
        self.loss = nn.CrossEntropyLoss(ignore_index=self.PAD_IDX, reduction="mean")

        # Initialize weights
        # ==================
        logger.info("Number of parameters: %s", self.count_parameters())
        logger.info("Initializing...")
        self.apply(self.init_weights)

    def _add_positional_embeddings(self, token_ids, embeddings):
        raise NotImplementedError

    def _make_attention_mask(self, target_len, context_len):
        attn_mask = torch.full(
            (target_len, context_len),
            fill_value=-float("inf"),
            dtype=torch.float,
        )
        # Example mask for context_len=10 and target_len=4
        # 0 0 0 0 0 0 0 - - -
        # 0 0 0 0 0 0 0 0 - -
        # 0 0 0 0 0 0 0 0 0 -
        # 0 0 0 0 0 0 0 0 0 0
        offset = context_len - target_len + 1
        attn_mask = torch.triu(attn_mask, diagonal=offset)
        return attn_mask

    def _predict(
        self,
        target_emb: torch.Tensor,
        context_emb: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ):
        # Construct attention mask
        # =========================
        context_len = context_emb.shape[1]
        target_len = target_emb.shape[1]
        attn_mask = self._make_attention_mask(target_len, context_len)

        # Run through the decoder
        # =======================
        decoded = self.decoder(
            target=target_emb,
            context=context_emb,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        logits = self.lm_head(decoded)  # shape (batch_size, seq_len, vocab_size)
        return logits

    def forward(
        self,
        tokens: TextFieldTensors,
    ) -> Dict[str, torch.Tensor]:
        # shape (batch_size, timesteps)
        token_ids = tokens["tokens"]["tokens"]

        # Get embeddings
        # ==============
        embeddings = self.embedder(tokens)
        embeddings = self._add_positional_embeddings(token_ids, embeddings)

        # Get source and target
        # =====================
        source = token_ids[:, :-1]
        source_emb = embeddings[:, :-1, :]
        if self.training:
            target = token_ids[:, 1:]  # shape: [B, N]
            target_emb = embeddings[:, 1:, :]  # shape: [B, N, D]
        else:
            target = token_ids[:, -1].unsqueeze(1)  # shape: [B, 1]
            target_emb = embeddings[:, -1, :].unsqueze(1)  # shape: [B, 1, D]

        # Invert the result because we want True to indicate pad
        key_mask = ~get_text_field_mask(source, padding_id=self.PAD_IDX)

        # Get logits
        # ==========
        logits = self._predict(
            target_emb=target_emb,
            context_emb=source_emb,
            key_padding_mask=key_mask,
        )

        # Calculate loss
        # ==============
        preds = logits.reshape(-1, self.vocab_size)
        reals = target.reshape(-1)

        loss = self.loss(preds, reals)
        self.metric(loss)
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
        output_dict["label"] = [
            self.vocab.get_token_from_index(x, namespace="tokens") for x in predicted_id
        ]
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"perplexity": self.metric.get_metric(reset)}

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

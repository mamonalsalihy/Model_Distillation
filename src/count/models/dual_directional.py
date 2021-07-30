# STL
import logging
import sys
import copy
from pathlib import Path
from typing import Dict, Optional

# Torch transformer
import torch
import torch.nn as nn

# AllenNLP
from allennlp.data import Vocabulary

# Models
from allennlp.models import Model
from allennlp.modules import Embedding
from allennlp.data import TensorDict
from allennlp.training.metrics import Perplexity

from src.count import config

sys.path.append(str(Path(__file__).resolve().parents[3]))

# Local
from src.count.decoders.base_decoder import Decoder
from src.count.models.base_transformer import Transformer

logger = logging.getLogger(__name__)


@Model.register("dual-directional-language-model", exist_ok=True)
class DualDirectionalModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        forward_model: Model,
        backward_model: Model,
        embedding_dim: int,
        forward_state_dict: Optional[str] = None,
        backward_state_dict: Optional[str] = None,
    ) -> None:
        super().__init__(vocab)

        self.forward_model = forward_model
        self.backward_model = backward_model
        self.embedding_dim = embedding_dim

        # Vocabulary stuff
        # ================
        self.vocab_size = vocab.get_vocab_size()
        self.PAD_IDX = self.vocab.get_token_index(config.PAD)

        # Evaluation
        # ==========
        self.perplexity = Perplexity()
        self.word_perplexity = Perplexity()
        self.loss = nn.CrossEntropyLoss(ignore_index=self.PAD_IDX, reduction="mean")

        # self.combined_lm_head = nn.Linear(2 * self.embedding_dim, self.vocab_size)

        if forward_state_dict is not None:
            state_dict = torch.load(forward_state_dict)
            self.forward_model.load_state_dict(state_dict)

        if backward_state_dict is not None:
            state_dict = torch.load(backward_state_dict)
            self.backward_model.load_state_dict(state_dict)

    def combine(self, forward, backward):
        S, B, V = forward.shape
        backward = torch.flip(backward, dims=[0])
        logits = torch.zeros_like(forward, device=forward.device)
        logits += forward
        logits[:-1] += backward
        logits[:-1] = logits[:-1] / 2
        return logits

    def combine_with_lm_head(self, forward, backward):
        backward = torch.flip(backward, dims=[0])  # flip along sequence axis

        # Align the backward embeddings to the forward ones
        backward_align = torch.zeros_like(forward)
        backward_align[:-1, :, :] = backward[1:, :, :]

        # Concatenate
        catted = torch.cat([forward, backward_align], dim=-1)  # [S, B, 2D]
        return self.combined_lm_head(catted)

    def encode(self, tokens):
        """Runs the input tokens through the decoder to get a contextual representation."""
        # Get forward & backward representations
        forward = self.forward_model.encode(tokens)
        backward = self.backward_model.encode(tokens)
        return self.combine(forward, backward)

    def forward(
        self,
        tokens: TensorDict,
        ratio: float,
    ) -> Dict[str, torch.Tensor]:
        labels = tokens.transpose(0, 1)[1:]  # [S, B]

        forward = self.forward_model(tokens, ratio)
        backward = self.backward_model(tokens, ratio)
        logits = self.combine(forward, backward)

        # Calculate loss
        # ==============
        preds = logits.reshape(-1, self.vocab_size)
        reals = labels.reshape(-1)
        loss = self.loss(preds, reals)

        self.perplexity(loss)
        self.word_perplexity(loss * ratio)

        # return combined logits & loss
        return {
            "logits": logits,
            "loss": loss,
        }

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
        return self.forward_model.make_output_human_readable(output_dict)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "perplexity": self.perplexity.get_metric(reset),
            "word_perplexity": self.word_perplexity.get_metric(reset),
        }

    def count_parameters(self):
        total = sum(p.numel() for p in self.forward_model.parameters() if p.requires_grad)
        total += sum(p.numel() for p in self.backward_model.parameters() if p.requires_grad)
        millions = total // 1_000_000
        thousands = (total - millions * 1_000_000) // 1_000
        string = str(millions) + "." + str(thousands) + "M"
        return string

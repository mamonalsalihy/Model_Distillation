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
        forward_state_dict: Optional[str] = None,
        backward_state_dict: Optional[str] = None,
    ) -> None:
        super().__init__(vocab)

        self.forward_model = forward_model
        self.backward_model = backward_model

        # Vocabulary stuff
        # ================
        self.vocab_size = vocab.get_vocab_size()
        self.PAD_IDX = self.vocab.get_token_index(config.PAD)

        # Evaluation
        # ==========
        self.metric = Perplexity()
        self.word_perplexity = Perplexity()
        self.loss = nn.CrossEntropyLoss(ignore_index=self.PAD_IDX, reduction="mean")

        if forward_state_dict is not None:
            state_dict = torch.load(forward_state_dict)
            self.forward_model.load_state_dict(state_dict)

        if backward_state_dict is not None:
            state_dict = torch.load(backward_state_dict)
            self.backward_model.load_state_dict(state_dict)

    def _forward_helper(self, tokens: TensorDict):
        pass

    def forward(
        self,
        tokens: TensorDict,
        ratio: float,
    ) -> Dict[str, torch.Tensor]:
        labels = tokens.transpose(0, 1)[1:]  # [S, B]

        forward = self.forward_model.forward(tokens, ratio)
        backward = self.backward_model.forward(tokens, ratio)

        # logits.size() = [seq len, batch_size, vocab len]
        forward_logits = forward["logits"]  # Logits for tokens 2 -> N
        backward_logits = torch.flip(backward["logits"], dims=[0])  # Logits for tokens 1 -> N-1

        # we don't need to consider the logits for the first token
        # we need to weight logits 2 -> N-1
        # logits for N don't need to be weighted
        B, Nm1, D = forward_logits.shape
        logits = torch.zeros(size=(B, Nm1, D), device=forward_logits.device, dtype=torch.float)  # 2 -> N

        logits[:-1, :, :] += forward_logits[:-1, :, :] / 2  # 2 -> N-1
        logits[-1, :, :] += forward_logits[-1, :, :]  # N
        logits[:-1, :, :] += backward_logits[1:, :, :] / 2  # 2 -> N-1

        # Calculate loss
        # ==============
        preds = logits.reshape(-1, self.vocab_size)
        reals = labels.reshape(-1)
        loss = self.loss(preds, reals)

        self.metric(loss)
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
            "perplexity": self.metric.get_metric(reset),
            "word_perplexity": self.word_perplexity.get_metric(reset),
        }

    def count_parameters(self):
        total = sum(p.numel() for p in self.forward_model.parameters() if p.requires_grad)
        total += sum(p.numel() for p in self.backward_model.parameters() if p.requires_grad)
        millions = total // 1_000_000
        thousands = (total - millions * 1_000_000) // 1_000
        string = str(millions) + "." + str(thousands) + "M"
        return string

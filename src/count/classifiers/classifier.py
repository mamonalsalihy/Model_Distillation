# STL
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Any, Callable
from overrides import overrides

# Torch
import torch
import torch.nn as nn

# AllenNLP
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.data.fields import MetadataField
from allennlp.modules import FeedForward
from allennlp.training.metrics import (
    SpearmanCorrelation,
    BooleanAccuracy,
    PearsonCorrelation,
    CategoricalAccuracy,
)
from allennlp.nn.util import get_text_field_mask


# Models
from allennlp.models import Model

try:
    sys.path.append(str(Path(__file__).resolve().parents[3]))
except NameError:
    sys.path.append(str(Path(".").resolve().parents[2]))


# Local
from src.count.classifiers.metrics import MCC  # type: ignore
from src.count.models.teacher_student import TeacherStudent  # type: ignore

logger = logging.getLogger(__name__)


@Model.register("glue-classifier", exist_ok=True)
class GLUEClassifier(Model):
    """Classifier for various GLUE tasks."""

    def __init__(
        self,
        vocab: Vocabulary,
        model: Model,
        embedding_dim: int,
        task: str,
        num_labels: int = 1,
        feedforward: Optional[FeedForward] = None,
        pool_method: str = "mean",
        **kwargs,
    ):
        super().__init__(vocab)

        self.metrics: Dict[str, Any] = {}
        if isinstance(model, TeacherStudent):
            self.model = model.student
        else:
            self.model = model
        self.feedforward = feedforward
        self.num_labels = num_labels
        self.embedding_dim = embedding_dim

        if pool_method == "sum":
            self.pooler = self.sum
        elif pool_method == "mean":
            self.pooler = self.mean
        elif pool_method == "max":
            self.pooler = self.max

        assert task in ["wnli", "stsb", "sst2", "cola"]
        self.task = task

        num_sents = 1

        if task == "wnli":
            self.num_labels = 3
            self.loss = nn.CrossEntropyLoss(reduction="mean")
            self.metrics = {"accuracy": CategoricalAccuracy()}
        elif task == "stsb":
            self.num_labels = 1
            self.loss = nn.MSELoss(reduction="mean")  # type: ignore
            self.metrics = {"pearsons": PearsonCorrelation(), "spearmans": SpearmanCorrelation()}
            num_sents = 2
        elif task == "sst2":
            self.num_labels = 1
            self.loss = nn.MSELoss(reduction="mean")  # type: ignore
            self.metrics = {"accuracy": CategoricalAccuracy()}
        elif task == "cola":
            self.num_labels = 2
            self.loss = nn.CrossEntropyLoss(reduction="mean")
            self.metrics = {"mcc": MCC()}

        self.forward_map: Dict[str, Callable] = {
            "cola": self._cola,
            "wnli": self._wnli,
            "sst2": self._sst2,
            "stsb": self._stsb,
        }

        self.head = nn.Linear(num_sents * self.embedding_dim, self.num_labels)

    @staticmethod
    def sum(seq):
        return torch.sum(seq, dim=1)

    @staticmethod
    def max(seq):
        return torch.max(seq, dim=1)[0]

    @staticmethod
    def mean(seq):
        return torch.mean(seq, dim=1)

    def encode_sentence(self, sentence):
        pad_mask = ~get_text_field_mask(sentence)  # don't forget to flip it with `~`!
        hidden = sentence["tokens"]["tokens"]

        hidden = self.model.encode(hidden, pad_mask, chop_off_last=False)
        hidden = hidden.transpose(0, 1)  # [B, S, D]
        if self.feedforward:
            hidden = self.feedforward(hidden)

        hidden = self.pooler(hidden)

        return hidden

    def _cola(self, tokens, idx, label):
        hidden = self.encode_sentence(tokens)

        # head
        logits = self.head(hidden)  # [B, 2]
        predictions = torch.argmax(logits, dim=-1)

        # loss & metrics
        if label.max() >= 0:
            loss = self.loss(logits, label)
            for k, met in self.metrics.items():
                met(predictions, label)
        else:
            loss = torch.tensor(0.0, device=hidden.device)

        return {
            "loss": loss,
            "logits": logits,
            "preds": predictions,
            "idx": idx,
        }

    def _wnli(self, tokens, idx, label):
        hidden = self.encode_sentence(tokens)

        # head
        logits = self.head(hidden)  # [B, 2]
        predictions = torch.argmax(logits, dim=-1)

        # loss & metrics
        if label.max() >= 0:
            loss = self.loss(logits, label)
            for k, met in self.metrics.items():
                met(predictions, label)
        else:
            loss = torch.tensor(0.0, device=hidden.device)

        return {
            "loss": loss,
            "logits": logits,
            "preds": predictions,
            "idx": idx,
        }

    def _sst2(self, tokens, idx, label):
        hidden = self.encode_sentence(tokens)

        # head
        logits = self.head(hidden)  # [B, 2]
        predictions = torch.argmax(logits, dim=-1)

        # loss & metrics
        if label.max() >= 0:
            loss = self.loss(logits, label)
            for k, met in self.metrics.items():
                met(predictions, label)
        else:
            loss = torch.tensor(0.0, device=hidden.device)

        return {
            "loss": loss,
            "logits": logits,
            "preds": predictions,
            "idx": idx,
        }

    def _stsb(self, one_two, two_one, idx, label):
        # encode both sentences
        s1 = self.encode_sentence(one_two)  # [B, D]
        s2 = self.encode_sentence(two_one)  # [B, D]

        # project into label space
        predictions = self.head(torch.cat([s1, s2], dim=-1))  # [B]

        # calculate loss & metrics
        if label.max() >= 0:
            loss = self.loss(predictions, label.view(-1, 1))
            for k, met in self.metrics.items():
                met(predictions, label.view(-1, 1))

        return {"loss": loss, "preds": predictions, "idx": idx}

    @overrides
    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        return self.forward_map[self.task](*args, **kwargs)

    def make_output_human_readable(self, output_dict):
        idxs = output_dict["idx"]
        preds = output_dict["preds"]

        lines = "\n".join([f"{int(i)+1},{p}" for i, p in zip(idxs, preds)])
        output_dict["lines"] = lines
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {k: v.get_metric(reset) for k, v in self.metrics.items()}

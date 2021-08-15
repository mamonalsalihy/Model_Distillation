# STL
import logging
import sys
from pathlib import Path
from typing import Dict, Optional
from overrides import overrides

# Torch
import torch
import torch.nn as nn

# AllenNLP
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.data.fields import MetadataField
from allennlp.modules import FeedForward
from allennlp.training.metrics import SpearmanCorrelation, BooleanAccuracy
from allennlp.nn.util import get_text_field_mask


# Models
from allennlp.models import Model

try:
    sys.path.append(str(Path(__file__).resolve().parents[3]))
except NameError:
    sys.path.append(str(Path(".").resolve().parents[2]))


# Local
from src.count.data import ColaReader, StsReader
from src.count.classifiers.metrics import MCC

logger = logging.getLogger(__name__)


@Model.register("cola-classifier", exist_ok=True)
class ColaClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        model: Model,
        embedding_dim: int,
        num_labels: int,
        feedforward: Optional[FeedForward] = None,
        pool_method: str = "mean",
        weights: torch.Tensor = None,
        **kwargs,
    ):
        "Classifier with custom metrics"
        super().__init__(vocab)

        self.model = model
        self.feedforward = feedforward
        self.num_labels = num_labels
        self.embedding_dim = embedding_dim
        self.classifier_head = nn.Linear(self.embedding_dim, self.num_labels)

        if pool_method == "sum":
            self.pooler = self.sum
        elif pool_method == "mean":
            self.pooler = self.mean
        elif pool_method == "max":
            self.pooler = self.max

        weights = weights or torch.ones(num_labels)
        self.loss = nn.CrossEntropyLoss(reduction="mean", weight=weights)
        self.accuracy = BooleanAccuracy()
        self.mcc = MCC()

    @staticmethod
    def sum(seq):
        return torch.sum(seq, dim=1)

    @staticmethod
    def max(seq):
        return torch.max(seq, dim=1)[0]

    @staticmethod
    def mean(seq):
        return torch.mean(seq, dim=1)

    @overrides
    def forward(
        self,
        tokens: TextFieldTensors,
        idx: MetadataField,
        label: torch.IntTensor = torch.IntTensor([-1]),
    ) -> Dict[str, torch.Tensor]:

        pad_mask = ~get_text_field_mask(tokens)  # don't forget to flip it with `~`!
        hidden = tokens["tokens"]["tokens"]

        B, S = hidden.shape

        hidden = self.model.encode(hidden, pad_mask)  # [S, B, D]
        hidden = hidden.transpose(0, 1)  # [B, S, D]

        # feedforward
        if self.feedforward:
            hidden = self.feedforward(hidden)

        # pool
        pooled = self.pooler(hidden)  # [B, D]

        # head
        logits = self.classifier_head(pooled)  # [B, 2]
        predictions = torch.argmax(logits, dim=-1)

        # loss & metrics
        if label.max() >= 0:
            loss = self.loss(logits, label)
            self.accuracy(predictions, label)
            self.mcc(predictions, label)
        else:
            loss = torch.tensor(0.0, device=hidden.device)

        return {
            "loss": loss,
            "logits": logits,
            "preds": predictions,
            "idx": idx,
        }

    def make_output_human_readable(self, output_dict):
        idxs = output_dict["idx"]
        preds = output_dict["preds"]

        lines = "\n".join([f"{int(i)+1},{p}" for i, p in zip(idxs, preds)])
        output_dict["lines"] = lines
        return output_dict

    def get_metrics(self, reset: bool = False):
        return {"accuracy": self.accuracy.get_metric(reset), "mcc": self.mcc.get_metric(reset)}


@Model.register("sts-classifier", exist_ok=True)
class StsClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        model: Model,
        embedding_dim: int,
        feedforward: Optional[FeedForward] = None,
        pool_method: str = "mean",
        **kwargs,
    ):
        "Classifier with custom metrics"
        super().__init__(vocab)

        self.model: Model = model
        if kwargs.get("freeze", False):
            self.model.eval()
        self.feedforward = feedforward
        self.embedding_dim = embedding_dim
        self.regression_head = nn.Linear(self.embedding_dim * 2, 1)

        if pool_method == "sum":
            self.pooler = self.sum
        elif pool_method == "mean":
            self.pooler = self.mean
        elif pool_method == "max":
            self.pooler = self.max
        elif pool_method == "cls":
            self.pooler = self.cls

        self.loss = nn.MSELoss(reduction="mean")
        self.spearman = SpearmanCorrelation()

    @staticmethod
    def sum(seq):
        return torch.sum(seq, dim=1)

    @staticmethod
    def max(seq):
        return torch.max(seq, dim=1)[0]

    @staticmethod
    def mean(seq):
        return torch.mean(seq, dim=1)

    @staticmethod
    def cls(seq):
        return seq[:, -1, :]

    def encode_sentence(self, sentence: TextFieldTensors):
        pad_mask = ~get_text_field_mask(sentence)  # don't forget to flip it with `~`!
        hidden = sentence["tokens"]["tokens"]

        hidden = self.model.encode(hidden, pad_mask, chop_off_last=False)
        hidden = hidden.transpose(0, 1)  # [B, S, D]
        if self.feedforward:
            hidden = self.feedforward(hidden)

        hidden = self.pooler(hidden)

        return hidden

    @staticmethod
    def scale_one_to_five(tensor: torch.Tensor):
        return (tensor + 1.0) * 5 / 2

    @overrides
    def forward(
        self,
        sentence1: TextFieldTensors,
        sentence2: TextFieldTensors,
        idx: torch.Tensor,
        label: torch.FloatTensor,
    ) -> Dict[str, torch.Tensor]:

        # encode both sentences
        s1 = self.encode_sentence(sentence1)  # [B, D]
        s2 = self.encode_sentence(sentence2)  # [B, D]

        # project into label space
        predictions = self.regression_head(torch.cat([s1, s2], dim=-1))  # [B]

        # clip values from 0 to 5
        # torch.clip(predictions, min=0, max=5)
        predictions = self.scale_one_to_five(predictions)

        # calculate loss & metrics
        if label.max() >= 0:
            loss = self.loss(predictions, label.view(-1, 1))
            self.spearman(predictions, label.view(-1, 1))

        return {"loss": loss, "preds": predictions, "idx": idx}

    def make_output_human_readable(self, output_dict):
        idxs = output_dict["idx"]
        preds = output_dict["preds"]

        lines = "\n".join([f"{int(i)+1},{p}" for i, p in zip(idxs, preds)])
        output_dict["lines"] = lines
        return output_dict

    def get_metrics(self, reset: bool = False):
        return {"spearman": self.spearman.get_metric(reset)}

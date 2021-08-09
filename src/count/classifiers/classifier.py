# STL
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, List

# Torch
import torch
import torch.nn as nn

# AllenNLP
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.data.fields import MetadataField
from allennlp.models import BasicClassifier
from allennlp.modules import Embedding, Seq2SeqEncoder, Seq2VecEncoder, FeedForward
from allennlp.nn import Activation, InitializerApplicator
from allennlp.modules.token_embedders import PassThroughTokenEmbedder
from allennlp.modules import TextFieldEmbedder
from allennlp.training.metrics import Metric, CategoricalAccuracy, BooleanAccuracy
from allennlp.nn.util import get_text_field_mask


# Models
from allennlp.models import Model

try:
    sys.path.append(str(Path(__file__).resolve().parents[3]))
except NameError:
    sys.path.append(str(Path(".").resolve().parents[2]))


# Local
from src.count import config
from src.count.data import ColaReader
from src.count.models.simple_transformer import SimpleTransformerLanguageModel
from src.count.classifiers.metrics import MCC

logger = logging.getLogger(__name__)


@Model.register("glue-classifier", exist_ok=True)
class GLUEClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        model: Model,
        embedding_dim: int,
        feedforward: Optional[FeedForward] = None,
        pool_method: str = "mean",
        num_labels: int = None,
        weights: List = None,
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

    def forward(
        self,
        tokens: TextFieldTensors,
        idx: MetadataField,
        label: torch.IntTensor = None,
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

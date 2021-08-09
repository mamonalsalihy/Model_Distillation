# STL
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

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
from allennlp.training.metrics import Metric, CategoricalAccuracy
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

logger = logging.getLogger(__name__)


@Model.register("glue-classifier", exist_ok=True)
class GLUEClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        model: Model,
        embedding_dim: int,
        feedforward: Optional[FeedForward] = None,
        pool_method: str = "sum",
        num_labels: int = None,
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

        self.loss = nn.BCEWithLogitsLoss(reduction='mean')


    @staticmethod
    def sum(self, seq):
        return torch.sum(seq, dim=1)

    @staticmethod
    def max(self, seq):
        return torch.max(seq, dim=1)[0]

    @staticmethod
    def mean(self, seq):
        return torch.mean(seq, dim=1)

    def forward(
        self,
        tokens: TextFieldTensors,
        label: torch.IntTensor = None,
        metadata: MetadataField = None,
    ) -> Dict[str, torch.Tensor]:

        pad_mask = get_text_field_mask(tokens)
        x = tokens['tokens']['tokens']

        B, S = x.shape

        # encode
        hidden = self.model.encode(x, pad_mask, chop_off_last=False)  # [S, B, D]
        hidden = hidden.transpose(0, 1)  # [B, S, D]

        # feedforward
        if self.feedforward:
            hidden = self.feedforward(hidden)

        # pool
        pooled = self.pooler(hidden) # [B, D]

        # head
        logits = self.classifier_head(hidden) # [B, 2]

        # loss & metrics
        loss = self.loss(logits, label)
        predictions = torch.argmax(logits, dim=-1)
        self.accuracy(predictions, label)

        return {
            "loss": loss,
            "logits": logits,
        }

    def get_metrics(self, reset: bool = False):
        return {"accuracy": self.accuracy.get_metric(reset)}



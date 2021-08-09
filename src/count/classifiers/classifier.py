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
from allennlp.models import BasicClassifier
from allennlp.modules import Embedding, Seq2SeqEncoder, Seq2VecEncoder, FeedForward
from allennlp.nn import Activation, InitializerApplicator
from allennlp.modules.token_embedders import PassThroughTokenEmbedder
from allennlp.modules import TextFieldEmbedder
from allennlp.training.metrics import Metric, CategoricalAccuracy


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
class GLUEClassifier(BasicClassifier):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        seq2seq_encoder: Optional[Seq2SeqEncoder] = None,
        feedforward: Optional[FeedForward] = None,
        metrics: Optional[Dict[str, Metric]] = None,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        namespace: str = "tokens",
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ):
        "Classifier with custom metrics"
        super().__init__(
            vocab,
            text_field_embedder,
            seq2vec_encoder,
            seq2seq_encoder,
            feedforward,
            dropout,
            num_labels,
            label_namespace,
            namespace,
            initializer,
            **kwargs,
        )

        self.metrics = metrics or {}
        self.metrics.update({"accuracy": self._accuracy})

    def forward(
        self,
        tokens: TextFieldTensors,
        label: torch.IntTensor = None,
        metadata: MetadataField = None,
    ) -> Dict[str, torch.Tensor]:
        output = super().forward(tokens, label, metadata)
        for metric in self.metrics.values():
            metric(output["logits"], label)
        return output

    def get_metrics(self, reset: bool = False):
        return {name: metric.get_metric(reset) for name, metric in self.metrics.items()}

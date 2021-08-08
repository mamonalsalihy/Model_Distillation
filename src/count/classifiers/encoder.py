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
from allennlp.models import BasicClassifier
from allennlp.modules import Embedding, Seq2SeqEncoder, Seq2VecEncoder, FeedForward
from allennlp.nn import Activation
from allennlp.modules.text_field_embedders import PassThroughTokenEmbedder
from allennlp.data.data_loaders import MultiProcessDataLoader


# Models
from allennlp.models import Model

sys.path.append(str(Path(__file__).resolve().parents[3]))

# Local
from src.count import config
from src.count.data import ColaReader
from src.count.models.base_transformer import Transformer
from src.count.models.simple_transformer import SimpleTransformerLanguageModel
from src.count.decoders.transformer_decoder import TransformerDecoder

logger = logging.getLogger(__name__)


class S2SEncoder(Seq2SeqEncoder):
    def __init__(self, model: Transformer):
        self.model = model

    def forward(self, tokens):
        """Encodes a sequence of `N` tokens into `N` embeddings.

        Arguments
        ---------
        tokens : torch.Tensor
            Input sequence to embed, of shape [B, S]

        Returns
        -------
        torch.Tensor :
            Sequence embedding of shape [B, D]
        """
        # encode -> pool -> return
        return self.model.encode(tokens)


@Seq2VecEncoder.register("glue-s2v-pooler")
class S2VEncoder(Seq2VecEncoder):
    def __init__(self, model: Transformer, pooler: str = "max"):
        self.model = model

        # Determine which pooling operation to use
        pooler = pooler.lower()
        if pooler == "max":
            self.pooler = S2VEncoder.max_pool
        elif pooler == "mean":
            self.pooler = S2VEncoder.mean_pool
        else:
            raise NotImplementedError(f"Pooling method {pooler} is not supported.")

    @staticmethod
    def max_pool(seq):
        return torch.max(seq, dim=0)

    @staticmethod
    def mean_pool(seq):
        return torch.mean(seq, dim=0)

    def forward(self, tokens):
        """Encodes a sequence of tokens into a single sequence embedding.

        Arguments
        ---------
        tokens : torch.Tensor
            Input sequence to embed, of shape [B, S]

        Returns
        -------
        torch.Tensor :
            Sequence embedding of shape [B, D]
        """
        # encode -> pool -> return
        seq_emb = self.model.encode(tokens)
        return self.pooler(seq_emb)  # [B, D]


if __name__ == "__main__":
    reader = ColaReader("../../wordpiece-tokenizer.json")

    loader = MultiProcessDataLoader(
        reader,
        data_path="train",
        batch_size=2,
    )
    vocab = Vocabulary.from_files(config.VOCAB_DIR, padding_token="[PAD]", oov_token="[UNK]")
    loader.index_with(vocab)

    model = SimpleTransformerLanguageModel.from_archive(
        "/data/users/nilay/the-count/saved-experiments/16M-model/model.tar.gz", vocab
    )
    embedder = PassThroughTokenEmbedder(vocab.get_vocab_size("tokens"))
    seq2vec = S2VEncoder(model)
    ff = FeedForward(
        input_dim=model.embedding_dim,
        num_layers=1,
        hidden_dims=4 * model.embedding_dim,
        activation=Activation.by_name("relu")(),
        dropout=0.2,
    )

    classifier = BasicClassifier(
        text_field_embedder=embedder,
        seq2vec_encoder=seq2vec,
        feedforward=ff,
        vocab=vocab,
        num_labels=vocab.get_vocab_size("labels"),
        dropout=0.2,
    )

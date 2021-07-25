# STL
import logging
from typing import Dict, Optional

# Torch
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# AllenNLP
from allennlp.data import TensorDict
from allennlp.data import Vocabulary

# Training
from allennlp.training.metrics import Perplexity

# Models
from allennlp.models import Model
from allennlp.modules import Embedding

from src.count import config
from src.count.decoders.base_decoder import Decoder

logger = logging.getLogger(__name__)


@Model.register("base-lstm", exist_ok=True)
class SimpleLSTMLanguageModel(Model):
    """
    This model contains a decoder which is just an LSTM module which takes as input an embedded representation of the text
    """

    def __init__(
            self,
            decoder: Decoder,
            embedder: Embedding,
            vocab: Vocabulary,
            hidden_dim: int,
            dropout: float
    ) -> None:
        super().__init__(vocab)
        # The decoder is an LSTM module
        self.decoder = decoder
        # The embedding matrix which takes the vectorized input and maps it to an embedded representation
        self.embedder = embedder
        # The hidden dimension of the LSTM. Typically four times the embedding dimension.
        self.hidden_dim = hidden_dim

        # The vocab size is the cardinality of the vocabulary
        self.vocab_size = vocab.get_vocab_size()
        self.PAD_INDX = vocab.get_token_index(config.PAD)

        # The language model head is a linear layer attached
        # to the end of the model for a consolidated representation of the input
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        # We create a dropout layer to prevent overfitting
        self.drop = nn.Dropout(dropout)

        # We initialize the weights of the language model head
        self.init_weights()

        # Metrics
        self.perplexity = Perplexity()
        self.word_perplexity = Perplexity()
        self.loss = nn.CrossEntropyLoss(ignore_index=self.PAD_INDX, reduction="mean")

        logger.info("Number of parameters: %s", self.count_parameters())

    def init_weights(self):
        nn.init.xavier_uniform_(self.lm_head.weight)

    def forward(
            self,
            tokens: TensorDict,
            sequence_len: list,
            ratio: float,
    ) -> Dict[str, torch.Tensor]:
        # Input representation for training.
        tokens = tokens.transpose(0, 1)
        source = tokens[:-1]
        labels = tokens[1:]

        # Transform input to embedded vectors
        emb = self.drop(self.embedder(source))
        # Sequence length list needs to be in decreasing order. Weird.
        sequence_len.sort(reverse=True)

        # Pack the embedded vectors for the LSTM
        packed_input = pack_padded_sequence(emb, sequence_len, batch_first=False)
        # Feed the packed tensors into the LSTM
        decoded, hidden = self.decoder(packed_input)
        # Unpack the tensors to feed into the linear layer
        unpacked_input, unpacked_lengths = pad_packed_sequence(decoded, padding_value=self.PAD_INDX, batch_first=False)

        logits = self.lm_head(unpacked_input)
        logits = self.drop(logits)

        preds = logits.reshape(-1, self.vocab_size)
        reals = labels.reshape(-1)

        # metrics and loss
        loss = self.loss(preds, reals)
        self.perplexity(loss)
        self.word_perplexity(loss * ratio)

        # Normalize to probabilities for the prediction.
        return {"logits": logits, "loss": loss}

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

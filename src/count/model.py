# STL
from typing import Dict

import numpy

# Utilities
import torch
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

# AllenNLP
from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer

# Models
from allennlp.models import Model
from allennlp.modules import Embedding, TextFieldEmbedder

# Inference
from allennlp.predictors.predictor import Predictor

# Training
from allennlp.training.metrics import Perplexity
from allennlp.training.trainer import GradientDescentTrainer, Trainer

# Layers
from allennlp.modules.attention import Attention
from allennlp.modules.transformer import TransformerLayer, TransformerStack
from allennlp.modules import Embedding, TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn.activations import Activation

# Torch transformer
import torch.nn as nn

# Local
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.count.data import WikiTextReader
from src.count import config


@Model.register("simple-transformer-language-model", exist_ok=True)
class SimpleTransformerLanguageModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        decoder: nn.Module,
        num_hidden_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        hidden_dropout: float = 0.2,
        activation: str = "relu",
    ) -> None:
        super().__init__(vocab)

        self.embedder = embedder
        self.activation = Activation.by_name(activation)()
        self.decoder = decoder

        # linear layer that maps the last last transformer layer to logits for each word
        self.vocab_size = vocab.get_vocab_size()
        self.PAD_IDX = self.vocab.get_token_index(config.PAD)
        self.linear = torch.nn.Linear(hidden_size, self.vocab_size)

        self.normalizer = config.BATCH_SIZE * config.CONTEXT_WINDOW
        self.dif_tokenizers_ratio = config.DIF_TOKENIZERS_RATIO

        self.metric = Perplexity()
        self.loss = nn.CrossEntropyLoss(ignore_index=self.PAD_IDX, reduction="mean")

    def forward(
        self,
        tokens: TextFieldTensors,
    ) -> Dict[str, torch.Tensor]:
        # shape (batch_size, timesteps)
        token_ids = tokens["tokens"]["tokens"]

        # Get source and target
        # =====================
        source = token_ids[:, :-1]
        target = token_ids[:, 1:]

        # Embed the tokens
        # ================
        # shape (batch_size, timesteps, embedding_size)
        embeddings = self.embedder(tokens)
        # get the first part of the window
        source_embeddings = embeddings[:, :-1, :]

        # Construct attention masks
        # =========================
        key_mask = get_text_field_mask(tokens, padding_id=self.PAD_IDX)[:, :-1].cuda()
        seq_len = source.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0).cuda()

        # Run through the decoder
        # =======================
        decoded = self.decoder(source_embeddings, attn_mask=mask, key_padding_mask=key_mask)
        logits = self.linear(decoded)  # shape (batch_size, seq_len, vocab_size)
        probs = torch.nn.functional.softmax(logits, dim=2)

        # reshape them because they aren't contiguous in memory
        # unsure why this issue exists in AllenNLP
        # https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107
        preds = logits.reshape(-1, self.vocab_size)
        target = target.reshape(-1)

        print(preds)

        # Calculate loss and normalize
        # ============================
        # temp = torch.nn.functional.cross_entropy(
        #     preds, target, ignore_index=self.PAD_IDX, reduction="sum"
        # )
        # loss = temp / self.normalizer
        # new_normalized = temp / (self.normalizer * self.dif_tokenizers_ratio)

        # just for testing
        loss = self.loss(preds, target)
        self.metric(loss)

        return {"logits": logits, "loss": loss, "probs": probs}

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
        # Take the logits from the forward pass, and compute the label
        # IDs for maximum values
        logits = output_dict["logits"].cpu().data.numpy()
        predicted_id: numpy.ndarray = numpy.argmax(logits, axis=-1)
        # Convert these IDs back to label strings using vocab
        output_dict["label"] = [
            self.vocab.get_token_from_index(x, namespace="tokens") for x in predicted_id
        ]
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"perplexity": self.metric.get_metric(reset)}

    # change parameters to be a more readable format
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        millions = total // 1_000_000
        thousands = (total - millions * 1_000_000) // 1_000
        string = str(millions) + "." + str(thousands) + "M"
        return string

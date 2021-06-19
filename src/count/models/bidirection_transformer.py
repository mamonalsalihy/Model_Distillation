# STL
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy
import torch

# Torch transformer
import torch.nn as nn

# AllenNLP
from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer

# Models
from allennlp.models import Model
from allennlp.modules import Embedding, TextFieldEmbedder

# Layers
from allennlp.modules.attention import Attention
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.transformer import TransformerLayer, TransformerStack
from allennlp.modules.transformer.positional_encoding import SinusoidalPositionalEncoding
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.initializers import InitializerApplicator

# Inference
from allennlp.predictors.predictor import Predictor

# Training
from allennlp.training.metrics import Perplexity
from allennlp.training.trainer import GradientDescentTrainer, Trainer

sys.path.append(str(Path(__file__).resolve().parents[3]))

# Local
from src.count import config
from src.count.data import WikiTextReader
from src.count.decoders.base_decoder import Decoder

logger = logging.getLogger(__name__)


@Model.register("bidirection-transformer-language-model", exist_ok=True)
class BiDirectionTransformerLanguageModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        decoder: Decoder,
        embedding_dim: int,
        max_positions: int,
    ) -> None:
        super().__init__(vocab)

        self.embedder = embedder
        self.pos_embedder = nn.Embedding(max_positions, embedding_dim)
        self.decoder = decoder

        # linear layer that maps the last last transformer layer to logits for each word
        self.vocab_size = vocab.get_vocab_size()
        self.PAD_IDX = self.vocab.get_token_index(config.PAD)
        self.lm_head = torch.nn.Linear(embedding_dim, self.vocab_size, bias=False)
        self.lm_head.weight = self.embedder._token_embedders["tokens"].weight

        # self.normalizer = config.BATCH_SIZE * config.CONTEXT_WINDOW
        # self.dif_tokenizers_ratio = config.DIF_TOKENIZERS_RATIO

        self.metric = Perplexity()
        self.loss = nn.CrossEntropyLoss(ignore_index=self.PAD_IDX, reduction="mean")
        logger.info("Number of parameters: %s", self.count_parameters())

        # Initialize weights
        logger.info("Initializing...")
        self.apply(self.init_weights)

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
        positions = torch.arange(token_ids.shape[1], device=embeddings.device).unsqueeze(-1)
        pos_embeddings = self.pos_embedder(positions).permute(1, 0, 2).expand_as(embeddings)
        embeddings = embeddings + pos_embeddings

        # get the first part of the window
        source_embeddings = embeddings[:, :-1, :]

        # Construct attention masks
        # =========================
        key_mask = get_text_field_mask(tokens, padding_id=self.PAD_IDX)[:, :-1]
        # invert the mask, so we have True where padding is and False where words are
        key_mask = ~key_mask

        # the only token we want to mask is the one we are predicting
        seq_len = source.shape[1]
        mask_values = torch.full(
            (seq_len-1,),
            fill_value=-float("inf"),
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        attn_mask = torch.diag(mask_values, diagonal=1)

        # Run through the decoder
        # =======================
        decoded = self.decoder(source_embeddings, attn_mask=attn_mask, key_padding_mask=key_mask)
        logits = self.lm_head(decoded)  # shape (batch_size, seq_len, vocab_size)
        probs = torch.nn.functional.softmax(logits, dim=2)

        # Calculate loss and normalize
        # ============================
        # temp = torch.nn.functional.cross_entropy(
        #     preds, target, ignore_index=self.PAD_IDX, reduction="sum"
        # )
        # loss = temp / self.normalizer
        # new_normalized = temp / (self.normalizer * self.dif_tokenizers_ratio)

        preds = logits.reshape(-1, self.vocab_size)
        target = target.reshape(-1)
        # if self.training:
        #     preds = logits.reshape(-1, self.vocab_size)
        #     target = target.reshape(-1)
        # else:  # If we're evaluating, we only care about the last prediction
        #     logits = logits[:, -1, :]
        #     probs = probs[:, -1, :]
        #     preds = logits.reshape(-1, self.vocab_size)
        #     target = target[:, -1].reshape(-1)

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

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        millions = total // 1_000_000
        thousands = (total - millions * 1_000_000) // 1_000
        string = str(millions) + "." + str(thousands) + "M"
        return string

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()
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


@Model.register("student-language-model", exist_ok=True)
class StudentModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        decoder: Decoder,
        teacher: Model,
        hidden_size: int,
        initializer: InitializerApplicator = None,
    ) -> None:
        super().__init__(vocab)

        self.embedder = embedder
        self.pos_emb = SinusoidalPositionalEncoding()
        self.decoder = decoder

        # linear layer that maps the last last transformer layer to logits for each word
        self.vocab_size = vocab.get_vocab_size()
        self.PAD_IDX = self.vocab.get_token_index(config.PAD)
        self.linear = torch.nn.Linear(hidden_size, self.vocab_size)

        self.normalizer = config.BATCH_SIZE * config.CONTEXT_WINDOW
        self.dif_tokenizers_ratio = config.DIF_TOKENIZERS_RATIO

        self.metric = Perplexity()
        self.kl_div = nn.KLDivLoss(reduction="mean")
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=self.PAD_IDX, reduction="mean")

        self.teacher = teacher
        # does this work? will passing the model to the trainer set this to train?
        self.teacher.eval()  # we don't want to train the teacher
        logger.info("Number of parameters: %s", self.count_parameters())

        # Initialize weights
        if initializer is not None:
            initializer(self)

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

        # add the positional information to the emeddings
        emb_pos = self.pos_emb(embeddings)

        # get the first part of the window
        source_embeddings = emb_pos[:, :-1, :]

        # Construct attention masks
        # =========================
        key_mask = get_text_field_mask(tokens, padding_id=self.PAD_IDX)[:, :-1].cuda()
        # invert the mask, so we have True where padding is and False where words are
        key_mask = ~key_mask
        seq_len = source.shape[1]
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0).cuda()

        # Run through the decoder
        # =======================
        decoded = self.decoder(source_embeddings, attn_mask=mask, key_padding_mask=key_mask)
        logits = self.linear(decoded)  # shape (batch_size, seq_len, vocab_size)
        # KLDivergence expects log probabilities for the student probabilities
        student_probs = torch.nn.functional.log_softmax(logits, dim=2)

        # Calculate the teacher's logits
        # ==============================
        if self.training:
            with torch.no_grad():
                teacher_output = self.teacher(tokens)
                soft_labels = teacher_output["logits"]
                # KLDivergence expects probabilities for the teacher tensor
                teacher_probs = torch.nn.functional.softmax(soft_labels, dim=2)

        # Calculate loss & Perplexity
        # ===========================

        # Perplexity
        # perplexity is calculated using the output of the student and the golen truth
        if self.training:
            preds = logits.reshape(-1, self.vocab_size)
            target = target.reshape(-1)

        else:  # If we're evaluating, we only care about the last prediction
            logits = logits[:, -1, :]
            student_probs = student_probs[:, -1, :]
            preds = logits.reshape(-1, self.vocab_size)
            target = target[:, -1].reshape(-1)

        student_loss = self.cross_entropy(preds, target)
        self.metric(student_loss)

        # Loss - If we're training, use kl_div. If we're evaluating, use CE
        if self.training:
            loss = self.kl_div(student_probs, teacher_probs)
        else:
            loss = student_loss

        logger.info("Student Loss: %s", student_loss.item)
        logger.info("KLDivergence Loss x 1e10: %s", loss.item * 1e10)

        return {"logits": logits, "loss": loss, "log_probs": student_probs, "student_loss": student_loss}

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
        teacher_parameters = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad)
        total -= teacher_parameters
        millions = total // 1_000_000
        thousands = (total - millions * 1_000_000) // 1_000
        string = str(millions) + "." + str(thousands) + "M"
        return string

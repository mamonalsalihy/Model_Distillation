# STL
from typing import Dict

import numpy

# Utilities
import torch

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
from allennlp.modules.transformer import TransformerLayer
from allennlp.nn.activations import Activation
from allennlp.nn.util import get_text_field_mask

# Training
from allennlp.training.metrics import Perplexity

# Local
from data import WikiTextReader


@Model.register("language-model")
class LanguageModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        att_dropout: float = 0.2,
        hidden_dropout: float = 0.2,
        activation: str = "relu",
        cross_attention: bool = False,
    ) -> None:
        super().__init__(vocab)

        self.embedder = embedder
        self.activation = Activation.by_name(activation)()
        self.transformer = TransformerLayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=att_dropout,
            hidden_dropout=hidden_dropout,
            activation=self.activation,
            add_cross_attention=cross_attention,
        )
        self.metric = Perplexity()

    def forward(
        self,
        tokens: TextFieldTensors,
        target: TextFieldTensors,
    ) -> Dict[str, torch.Tensor]:

        # do embedding stuff here
        # shape (batch_size, timesteps, embedding_size)
        embeddings = self.embedder(tokens)

        # do processing stuff here
        # print(f"tokens: {tokens}")
        # print(f"target: {target}")
        mask = get_text_field_mask(tokens)
        # return logits, maybe calculate loss?
        logits = self.transformer(embeddings, mask)
        # probs = torch.softmax(logits, dim=1)

        self.metric(logits, target)
        return {"logits": logits}

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


if __name__ == "__main__":
    reader = WikiTextReader(100)
    instances = list(reader.read("../data/wikitext-103/wiki.mini.tokens"))[:5]
    # generates a vocabulary from the file
    vocab = Vocabulary.from_instances(instances)
    # creates an embedder, needs the number of items in the vocab
    embedding = Embedding(num_embeddings=vocab.get_vocab_size(), embedding_dim=20)
    embedder = BasicTextFieldEmbedder(token_embedders={"tokens": embedding})
    data_loader = SimpleDataLoader(instances, batch_size=2, vocab=vocab)

    model = LanguageModel(
        vocab=vocab,
        embedder=embedder,
        hidden_size=20,
        intermediate_size=50,
        num_attention_heads=1,
    )
    for i, batch in zip(range(2), data_loader):
        model(**batch)

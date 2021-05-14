# STL
from typing import Dict

# Utilities
import torch
import numpy
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

# AllenNLP
from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

# Models
from allennlp.models import Model

# Inference
from allennlp.predictors.predictor import Predictor

# Training
from allennlp.training.metrics import Perplexity
from allennlp.training.trainer import GradientDescentTrainer, Trainer

# Layers
from allennlp.modules.attention import Attention
from allennlp.modules.transformer import TransformerLayer
from allennlp.modules import Embedding, TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn.activations import Activation

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
        # question: what is intermediate size?
        self.transformer = TransformerLayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=att_dropout,
            hidden_dropout=hidden_dropout,
            activation=self.activation,
            add_cross_attention=cross_attention,
        )

        # linear layer that maps the last last transformer layer to logits for each word
        self.vocab_size = vocab.get_vocab_size()
        self.linear = torch.nn.Linear(hidden_size, self.vocab_size)

        self.metric = Perplexity()

    def forward(
        self,
        tokens: TextFieldTensors,
        target: TextFieldTensors,
    ) -> Dict[str, torch.Tensor]:

        # shape (batch_size, timesteps)
        token_ids = tokens['tokens']['tokens']

        # get source and targets from tokens
        source = token_ids[:, 0:-1]
        target = token_ids[:, 1:]

        # do embedding stuff here
        # shape (batch_size, timesteps, embedding_size)
        embeddings = self.embedder(tokens)

        # get the first part of the window
        source_embeddings = embeddings[:, 0:-1, :]
        # do processing stuff here
        mask = get_text_field_mask(tokens)[:, 0:-1]
        # calculate logits of the next context
        trans_out = self.transformer(source_embeddings, mask)[0]

        # shape (batch_size, timesteps, vocab_size)
        logits = self.linear(trans_out)

        probs = torch.nn.functional.softmax(logits, dim=2)

        # reshape them because they aren't contiguous in memory
        # unsure why this issue exists in AllenNLP
        # https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107
        preds = logits.reshape(-1, self.vocab_size)
        target = target.reshape(-1)

        # need to pass pad idx so we can ignore this, unsure how to achieve this in AllenNLP
        loss = torch.nn.functional.cross_entropy(preds, target)

        # calculates the perplexity for the model
        self.metric(loss)

        return {"logits": logits, "loss": loss, 'probs': probs}

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
    instances = list(reader.read("../data/wikitext-103/wiki.mini.tokens"))
    # generates a vocabulary from the file
    vocab = Vocabulary.from_instances(instances)
    # creates an embedder, needs the number of items in the vocab
    embedding = Embedding(num_embeddings=vocab.get_vocab_size(), embedding_dim=20)
    embedder = BasicTextFieldEmbedder(token_embedders={"tokens": embedding})
    data_loader = SimpleDataLoader(instances, batch_size=4, vocab=vocab)

    model = LanguageModel(
        vocab=vocab,
        embedder=embedder,
        hidden_size=20,
        intermediate_size=50,
        num_attention_heads=1,
    )

    trainer = GradientDescentTrainer(
        model=model.cuda(),
        data_loader=data_loader,
        num_epochs=5,
        optimizer=torch.optim.Adam(model.parameters()),
    )

    trainer.train()

    pred = Predictor(model, data_loader)
    output = pred.predict_instance('I am a god.')
    print(output)

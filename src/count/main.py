# Utilities
import torch
import numpy
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from itertools import islice

# AllenNLP
from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

# Modules
from allennlp.modules import Embedding, TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

# Inference
from allennlp.predictors.predictor import Predictor

# Training
from allennlp.training.metrics import Perplexity
from allennlp.training.trainer import GradientDescentTrainer, Trainer

# Local
from data import WikiTextReader
import config
from model import LanguageModel

if __name__ == "__main__":
    # Build reader
    # ============
    reader = WikiTextReader(100)
    instances = reader.read(config.WIKI_RAW_DIR / "wiki.train.raw")

    # Read vocabulary from vocabulary directory
    # =========================================
    vocab = Vocabulary.from_files(config.VOCAB_DIR, padding_token=config.PAD, oov_token=config.UNK)

    # Create embedder for the model
    # =============================
    embedding = Embedding(num_embeddings=vocab.get_vocab_size(), embedding_dim=20)
    embedder = BasicTextFieldEmbedder(token_embedders={"tokens": embedding})

    # Setup model and training
    # ========================
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
        cuda_device=config.DEVICE_1,
    )

    # Run training
    # ============
    trainer.train()

    # pred = Predictor(model, data_loader)
    # output = pred.predict_instance("I am a god.")
    # print(output)

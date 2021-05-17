# Utilities
import torch
import numpy
from itertools import islice
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

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
    instances = list(islice(reader.read(config.WIKI_RAW_DIR / "wiki.train.raw"), 10))
    val_instances = list(islice(reader.read(config.WIKI_RAW_DIR / "wiki.valid.raw"), 10))

    # Read vocabulary from vocabulary directory
    # =========================================
    vocab = Vocabulary.from_files(config.VOCAB_DIR, padding_token=config.PAD, oov_token=config.UNK)

    # Create embedder for the model
    # =============================
    embedding = Embedding(num_embeddings=vocab.get_vocab_size(), embedding_dim=config.EMBEDDING_DIMENSION)
    embedder = BasicTextFieldEmbedder(token_embedders={"tokens": embedding})

    # Setup model and training
    # ========================
    data_loader = SimpleDataLoader(instances, batch_size=config.BATCH_SIZE, vocab=vocab)
    val_data_loader = SimpleDataLoader(val_instances, batch_size=config.BATCH_SIZE, vocab=vocab)

    model = LanguageModel(
        vocab=vocab,
        embedder=embedder,
        hidden_size=config.EMBEDDING_DIMENSION,
        intermediate_size=config.HIDDEN_DIMENSION,
        num_attention_heads=config.NUM_ATTENTION_HEADS,
    )

    trainer = GradientDescentTrainer(
        model=model.cuda(),
        data_loader=data_loader,
        validation_metric='-perplexity',
        validation_data_loader=val_data_loader,
        num_epochs=config.NUM_EPOCHS,
        patience=config.PATIENCE,
        optimizer=torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE),
        cuda_device=config.DEVICE_1,
    )

    # note, count_parmeters now returns a string for easier readability
    print('parameters:', model.count_parameters())

    # Run training
    # ============
    trainer.train()

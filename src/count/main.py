# Utilities
import numpy as np
import torch
import numpy
from itertools import islice
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
import os
import logging

# AllenNLP
from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.data_loaders import SimpleDataLoader, MultiProcessDataLoader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer

# Modules
from allennlp.modules import Embedding, TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

# Inference
from allennlp.predictors.predictor import Predictor

# Training
from allennlp.training.metrics import Perplexity
from allennlp.training.trainer import GradientDescentTrainer, Trainer

# Local
import config
from data import WikiTextReader
from tokenizer import WikiTextTokenizer
from model import LanguageModel

import sys

sys.path.append("../")
from src.utils.misc_utils import get_model_size
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # Build tokenizer
    # ===============
    wiki_tokenizer = WikiTextTokenizer(
        tokenizer_path=config.TOKENIZER,
        add_special_tokens=True,
    )

    # Build reader
    # ============
    reader = WikiTextReader(
        context=config.CONTEXT_WINDOW,
        tokenizer=wiki_tokenizer,
        token_indexers={"tokens": SingleIdTokenIndexer(namespace="tokens")},
        manual_distributed_sharding=True,
        manual_multiprocess_sharding=True,
        max_instances=config.MAX_INSTANCES,
    )

    # Read vocabulary from vocabulary directory
    # =========================================
    vocab = Vocabulary.from_files(config.VOCAB_DIR, padding_token=config.PAD, oov_token=config.UNK)

    # Create embedder for the model
    # =============================
    embedding = Embedding(num_embeddings=vocab.get_vocab_size(), embedding_dim=config.EMBEDDING_DIMENSION)
    embedder = BasicTextFieldEmbedder(token_embedders={"tokens": embedding})

    # Setup model and training
    # ========================
    train_data_loader = MultiProcessDataLoader(
        reader=reader,
        data_path=os.path.join(config.WIKI_RAW_DIR, "wiki.train.raw"),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        max_instances_in_memory=None,
        num_workers=4,
        start_method='spawn'
    )
    train_data_loader.index_with(vocab)
    val_data_loader = MultiProcessDataLoader(
        reader=reader,
        data_path=os.path.join(config.WIKI_RAW_DIR, "wiki.valid.raw"),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        max_instances_in_memory=None,
        num_workers=4,
        start_method='spawn'
    )
    val_data_loader.index_with(vocab)

    model = LanguageModel(
        vocab=vocab,
        embedder=embedder,
        num_hidden_layers=config.TRANSFORMER_LAYERS,
        hidden_size=config.EMBEDDING_DIMENSION,
        intermediate_size=config.HIDDEN_DIMENSION,
        num_attention_heads=config.NUM_ATTENTION_HEADS,
    )

    trainer = GradientDescentTrainer(
        model=model.to(config.DEVICE_1),
        data_loader=train_data_loader,
        validation_metric="-perplexity",
        validation_data_loader=val_data_loader,
        num_epochs=config.NUM_EPOCHS,
        patience=config.PATIENCE,
        optimizer=torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE),
        cuda_device=config.DEVICE_1,
    )

    # note, count_parmeters now returns a string for easier readability
    print('parameters:', model.count_parameters())
    print(get_model_size(model, saved=False))

    # Run training
    # ============
    trainer.train()

    # need some mechanism to save the model once its been trained

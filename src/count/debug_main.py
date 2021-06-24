# Utilities
import logging
import os
import sys
from itertools import islice
from pathlib import Path

import numpy
import numpy as np
import torch

# AllenNLP
from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.data_loaders import MultiProcessDataLoader, SimpleDataLoader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer

# Modules
from allennlp.modules import Embedding, TextFieldEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

# Inference
from allennlp.predictors.predictor import Predictor

# Training
from allennlp.training.metrics import Perplexity
from allennlp.training.trainer import GradientDescentTrainer, Trainer

import src.count.models.bidirection_transformer

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Local
from src.count import config
from src.count.data import WikiTextReader
from src.count.decoders.transformer_decoder import TransformerDecoder
from src.count.models.simple_transformer import SimpleTransformerLanguageModel
from src.count.models.direction_transformer import DirectionTransformerLanguageModel
from src.count.models.bidirection_transformer import BiDirectionTransformerLanguageModel
from src.count.models.student import StudentModel
from src.count.models.new_student import NewStudentModel
from src.count.tokenizer import WikiTextTokenizer
from src.utils.misc_utils import get_model_size


# sets the logging info to be very verbose
# logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)

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
        exclusive=True,
        split_on="sentence",
        min_context_len=2,
        manual_distributed_sharding=True,
        manual_multiprocess_sharding=True,
        max_instances=config.MAX_INSTANCES,
    )

    # Read vocabulary from vocabulary directory
    # =========================================
    vocab = Vocabulary.from_files(config.VOCAB_DIR, padding_token=config.PAD, oov_token=config.UNK)

    # Create embedder for the model
    # =============================
    embedding = Embedding(
        num_embeddings=vocab.get_vocab_size(), embedding_dim=config.EMBEDDING_DIMENSION
    )
    embedder = BasicTextFieldEmbedder(token_embedders={"tokens": embedding})

    # Setup model and training
    # ========================
    train_data_loader = MultiProcessDataLoader(
        reader=reader,
        data_path=os.path.join(config.WIKI_RAW_DIR, "wiki.train.raw"),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        max_instances_in_memory=config.MAX_INSTANCES_IN_MEMORY,
        num_workers=4,
        start_method="spawn",
    )
    train_data_loader.index_with(vocab)
    val_data_loader = MultiProcessDataLoader(
        reader=reader,
        data_path=os.path.join(config.WIKI_RAW_DIR, "wiki.valid.raw"),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        max_instances_in_memory=config.MAX_INSTANCES_IN_MEMORY,
        num_workers=4,
        start_method="spawn",
    )
    val_data_loader.index_with(vocab)

    # Make our custom decoder
    # =======================
    decoder = TransformerDecoder(
        input_dim=config.EMBEDDING_DIMENSION,
        hidden_dim=config.HIDDEN_DIMENSION,
        num_attention_heads=config.NUM_ATTENTION_HEADS,
        num_layers=config.TRANSFORMER_LAYERS,
        dropout=config.DROPOUT,
        activation=config.ACTIVATION,
    )

    model = BiDirectionTransformerLanguageModel(
        vocab=vocab,
        embedder=embedder,
        decoder=decoder.to(config.DEVICE_1),
        embedding_dim=config.EMBEDDING_DIMENSION,
        max_positions=config.CONTEXT_WINDOW,
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
    print("parameters:", model.count_parameters())
    print(get_model_size(model, saved=False))

    # Run training
    # ============
    trainer.train()
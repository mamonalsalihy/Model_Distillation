# Utilities
import numpy as np
import torch
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
import os

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
        max_instances=30_000,
    )

    # Read vocabulary from vocabulary directory
    # =========================================
    vocab = Vocabulary.from_files(config.VOCAB_DIR, padding_token=config.PAD, oov_token=config.UNK)

    # Create embedder for the model
    # =============================
    embedding = Embedding(num_embeddings=vocab.get_vocab_size(), embedding_dim=20)
    embedder = BasicTextFieldEmbedder(token_embedders={"tokens": embedding})

    # Setup model and training
    # ========================
    train_data_loader = MultiProcessDataLoader(
        reader=reader,
        data_path=os.path.join(config.WIKI_RAW_DIR , "wiki.train.raw"),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        max_instances_in_memory=None,
        num_workers=4,
    )
    train_data_loader.index_with(vocab)
    val_data_loader = MultiProcessDataLoader(
        reader=reader,
        data_path=os.path.join(config.WIKI_RAW_DIR , "wiki.valid.raw"),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        max_instances_in_memory=None,
        num_workers=4,
    )
    val_data_loader.index_with(vocab)

    model = LanguageModel(
        vocab=vocab,
        embedder=embedder,
        hidden_size=20,
        intermediate_size=50,
        num_attention_heads=1,
    )

    trainer = GradientDescentTrainer(
        model=model.to(config.DEVICE_1),
        data_loader=train_data_loader,
        validation_metric="-perplexity",
        validation_data_loader=val_data_loader,
        num_epochs=20,
        patience=10,
        optimizer=torch.optim.Adam(model.parameters()),
        cuda_device=config.DEVICE_1,
    )

    print("parameters:", model.count_parameters())

    # Run training
    # ============
    trainer.train()

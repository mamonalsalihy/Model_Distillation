""" Project configuration stuff, e.g. filepaths, special tokens, etc.

Hyperparameters should go in the AllenNLP config file, whenever we manage to get that working.
"""

import os
from pathlib import Path

import torch

# Project root
ROOT = str(Path(__file__).resolve().parents[2])

# Paths
TOKENIZER = os.path.join(ROOT, "unigram-tokenizer.json")
DATA_DIR = os.path.join(ROOT, "data")
VOCAB_DIR = os.path.join(ROOT, "data", "vocab")
WIKI_RAW_DIR = os.path.join(ROOT, "data", "wikitext-103-raw")
SRC_DIR = os.path.join(ROOT, "src")

# Parameters
VOCAB_SIZE = 30_000
DEVICE_1 = 0 if torch.cuda.is_available() else -1

# Transformer Parameters
TRANSFORMER_LAYERS = 4
EMBEDDING_DIMENSION = 32
HIDDEN_DIMENSION = 64
NUM_ATTENTION_HEADS = 4
ACTIVATION = "relu"

# Hyper-parameters
CONTEXT_WINDOW = 64

# Training Parameters
NUM_EPOCHS = 50
LEARNING_RATE = 2.5e-4
BATCH_SIZE = 32
PATIENCE = 10
MAX_INSTANCES = None
MAX_INSTANCES_IN_MEMORY = 4_000
DROPOUT = 0.2

# Special tokens
PAD = "[PAD]"
UNK = "[UNK]"
CLS = "[CLS]"
SEP = "[SEP]"

# Constants
# Currently the denominator is arbitrary. It should be the total number of tokens from the new tokenizer.
# Support for subset training has yet to be added.
DIF_TOKENIZERS_RATIO = 101_425_671 / 111_568_238

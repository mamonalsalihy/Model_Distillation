""" Project configuration stuff, e.g. filepaths, special tokens, etc.

Hyperparameters should go in the AllenNLP config file, whenever we manage to get that working.
"""

from pathlib import Path
import torch
import os

# Project root
ROOT = str(Path(__file__).resolve().parents[2])

# Paths
TOKENIZER = os.path.join(ROOT, "unigram-tokenizer.json")
DATA_DIR = os.path.join(ROOT, "data")
VOCAB_DIR = os.path.join(ROOT, "data", "vocab")
WIKI_RAW_DIR = os.path.join(ROOT, "data", "wikitext-103-raw")

# Parameters
VOCAB_SIZE = 30_000
DEVICE_1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transformer Parameters
TRANSFORMER_LAYERS = 1
EMBEDDING_DIMENSION = 8
HIDDEN_DIMENSION = 16
NUM_ATTENTION_HEADS = 1
ACTIVATION = "relu"

# Hyper-parameters
CONTEXT_WINDOW = 10

# Training Parameters
NUM_EPOCHS = 1
LEARNING_RATE = 1e-5
BATCH_SIZE = 4
PATIENCE = 10
MAX_INSTANCES = 100
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

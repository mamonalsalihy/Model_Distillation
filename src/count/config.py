""" Project configuration stuff, e.g. filepaths, special tokens, etc.

Hyperparameters should go in the AllenNLP config file, whenever we manage to get that working.
"""

from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[2]

# Paths
TOKENIZER = ROOT / "unigram-tokenizer.json"
DATA_DIR = ROOT / "data"
VOCAB_DIR = ROOT / "data" / "vocab"
WIKI_RAW_DIR = ROOT / "data" / "wikitext-103-raw"

# Parameters
VOCAB_SIZE = 30_000
DEVICE_1 = 0  # device number

# Transformer Parameters
TRANSFORMER_LAYERS = 4
EMBEDDING_DIMENSION = 64
HIDDEN_DIMENSION = 128
NUM_ATTENTION_HEADS = 4

# Training Parameters
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
BATCH_SIZE = 4
PATIENCE = 10

# Special tokens
PAD = "[PAD]"
UNK = "[UNK]"
CLS = "[CLS]"
SEP = "[SEP]"

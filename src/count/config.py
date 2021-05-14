""" Project configuration stuff, e.g. filepaths, special tokens, etc.

Hyperparameters should go in the AllenNLP config file, whenever we manage to get that working.
"""

from pathlib import Path
import torch

# Project root
ROOT = Path(__file__).resolve().parents[2]

# Paths
TOKENIZER = ROOT / "unigram-tokenizer.json"
DATA_DIR = ROOT / "data"
VOCAB_DIR = ROOT / "data" / "vocab"
WIKI_RAW_DIR = ROOT / "data" / "wikitext-103-raw"

# Parameters
VOCAB_SIZE = 30_000
DEVICE_1 = "cuda" if torch.cuda.is_available() else "cpu"

# Special tokens
PAD = "[PAD]"
UNK = "[UNK]"
CLS = "[CLS]"
SEP = "[SEP]"

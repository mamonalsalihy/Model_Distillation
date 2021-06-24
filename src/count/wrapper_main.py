# Utilities
import logging
import os
import sys
from itertools import islice
from pathlib import Path

import numpy
import numpy as np
import torch
import shutil

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

# CLI
from allennlp.commands import main


sys.path.append(str(Path(__file__).resolve().parents[2]))

# Local
from src.count import config
from src.count.data import WikiTextReader
from src.count.decoders.transformer_decoder import TransformerDecoder
from src.count.models.simple_transformer import SimpleTransformerLanguageModel
from src.count.models.student import StudentModel
from src.count.models.new_student import NewStudentModel
from src.count.tokenizer import WikiTextTokenizer
from src.utils.misc_utils import get_model_size


# logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)

if __name__ == "__main__":
    config_file = "../configs/student/new-student.jsonnet"

    serialization_dir = "experiments/student_test"

    # Training will fail if the serialization directory already
    # has stuff in it. If you are running the same training loop
    # over and over again for debugging purposes, it will.
    # Hence we wipe it out in advance.
    # BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
    shutil.rmtree(serialization_dir, ignore_errors=True)

    # Assemble the command into sys.argv
    sys.argv = [
        "allennlp",  # command name, not used by main
        "train",
        config_file,
        "-s", serialization_dir,
        "--include-package", "src.count",
    ]

    main()
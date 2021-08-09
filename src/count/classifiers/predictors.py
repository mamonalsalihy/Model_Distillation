# STL
import logging
import sys
from typing import List, Dict, Optional
from argparse import ArgumentParser
from pathlib import Path

# Torch
import torch
import torch.nn as nn

# AllenNLP
from allennlp.data import Vocabulary
from allennlp.models import BasicClassifier
from allennlp.modules import Embedding, Seq2SeqEncoder, Seq2VecEncoder, FeedForward
from allennlp.nn import Activation
from allennlp.modules.token_embedders import PassThroughTokenEmbedder
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data import DatasetReader


# Models
from allennlp.models import Model

sys.path.append(str(Path(__file__).resolve().parents[3]))

# Local
from src.count import config
from src.count.data import ColaReader
from src.count.classifiers.classifier import GLUEClassifier
from src.count.models.simple_transformer import SimpleTransformerLanguageModel

logger = logging.getLogger(__name__)


class GLUEWriter:
    def __init__(self, model: Model, dataset_reader: DatasetReader, output_file: str):
        """Predictor to get GLUE things"""
        self.model = model
        self.dataset_reader = dataset_reader
        self.output_file = output_file

    def run(self, split):
        loader = MultiProcessDataLoader(
            self.dataset_reader,
            data_path=split,
            batch_size=32,
        )
        loader.index_with(self.model.vocab)
        lines = []
        for batch in loader:
            outputs = self.model.forward_on_instances(batch)
            lines.append(outputs["lines"])

        with open(self.output_file, "w") as f:
            f.writelines(lines)

        return lines


if __name__ == "__main__":
    parser = ArgumentParser("make predictions and dump to file")
    parser.add_argument("--archive_file", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_split", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    args = parser.parse_args()

    vocab = Vocabulary.from_files(config.VOCAB_DIR, padding_token="[PAD]", oov_token="[UNK]")

    model = GLUEClassifier.from_archive(args.archive_file)

    if args.dataset == "cola":
        dataset_reader = ColaReader(args.tokenizer)
    else:
        raise ValueError(f"Dataset {args.dataset} isn't supported yet!")

    writer = GLUEWriter(model, dataset_reader, args.output_file)
    writer.run(args.dataset_split)

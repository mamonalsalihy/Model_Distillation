# STL
import sys
import logging
import math
import os
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from tqdm import tqdm

# Torch
import torch

# AllenNLP
from allennlp.data import Vocabulary
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

# Data types
from allennlp.data.fields import Field, TextField, FlagField, TensorField, MetadataField
from allennlp.data.instance import Instance

# Indexers
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data import Token

# Tokenizers
from tokenizers import Tokenizer

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Local
from src.count import config
from src.count.tokenizer import WikiTextTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig()
logging.root.setLevel(logging.INFO)


class CachedReader(DatasetReader):
    def __init__(
        self,
        sequence_length: int,
        tokenizer_path: str,
        token_indexers: Dict[str, TokenIndexer] = None,
        exclusive: bool = True,
        lstm: bool = False,
        max_seq_len: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.token_indexers = token_indexers
        self.exclusive = exclusive
        self.lstm = lstm
        self.max_seq_len = max_seq_len

    def filter_lines(self, lines: List[str]) -> List[str]:
        raise NotImplementedError

    def _read(self, file_path: str) -> Iterable[Instance]:
        cache = Path(f"{file_path}.cache")
        dataset = {}
        if cache.is_file():
            logger.info("Loading cached dataset from %s", str(cache))
            dataset = torch.load(cache)
        else:
            logger.info(f"Reading data from {file_path}")
            with open(file_path, "r") as f:
                lines = [
                    line.strip() for line in tqdm(f) if line.strip() and line.strip()[0] != "="
                ]

            #  replace <unk>s with [UNK]s
            logger.info("Preprocessing...")
            lines = [line.replace("<unk>", "[UNK]") for line in tqdm(lines)]

            # tokenize
            logger.info("Generating tokens...")
            subwords = [self.tokenizer.encode(line).ids for line in tqdm(lines)]

            # flatten
            dataset["subwords"] = torch.tensor(
                [idx for line in subwords for idx in line], dtype=torch.long
            )

            logger.info("Generating words...")
            dataset["num_words"] = torch.tensor(
                [len(line.split(" ")) for line in lines], dtype=torch.long
            )

            logger.info("Caching tokenized dataset to %s", cache)
            with open(cache, "wb") as f:
                torch.save(dataset, f)

        # ratio to use when calculating word level perplexity
        ratio = len(dataset["subwords"]) / sum(dataset["num_words"])
        yield from self.tensor_to_instances(dataset["subwords"], ratio.item())

    def tensor_to_instances(self, subwords: torch.Tensor, ratio: float):
        logger.info("Building instances...")
        if self.lstm:
            eos_idx = self.tokenizer.token_to_id("[SEP]")
            sequence_indices = (subwords == eos_idx).nonzero()
            start = 0
            for i, end in enumerate(sequence_indices):
                seq = subwords[start : end + 1]
                if self.max_seq_len is None or seq.size(0) < self.max_seq_len:
                    yield Instance(
                        {
                            "tokens": TensorField(seq),
                            "sequence_len": MetadataField(len(seq) - 1),
                            "ratio": FlagField(ratio),
                        }
                    )
                start = end + 1
        elif self.exclusive:
            num_sequences = (subwords.size(0) // self.sequence_length) * self.sequence_length
            sequences = subwords.narrow(0, 0, num_sequences).view(-1, self.sequence_length)
            for inst in sequences:
                yield Instance({"tokens": TensorField(inst), "ratio": FlagField(ratio)})
        else:
            for end_idx in range(1, len(subwords)):
                start_idx = max(0, end_idx - self.sequence_length)
                inst = subwords[start_idx : end_idx + 1]
                yield Instance({"tokens": TensorField(inst), "ratio": FlagField(ratio)})

    def text_to_instance(
        self,
        text: str,
    ) -> Instance:
        tokens = torch.tensor(self.tokenizer(text).ids, dtype=torch.long)
        num_words = len(text.split())
        ratio = len(tokens) // num_words
        return Instance({"tokens": TensorField(tokens), "ratio": FlagField(ratio)})


@DatasetReader.register("wikitext-reader")
class WikiTextReader(CachedReader):
    def __init__(self, args):
        "docstring"

    @override
    def apply_token_indexers(self, instance) -> None:
        """Adds a token indexer to the instance. Automatically called by AllenNLP."""
        # instance["tokens"].token_indexers = self.token_indexers
        pass


if __name__ == "__main__":
    reader = WikiTextReader(
        sequence_length=4,
        tokenizer_path=config.TOKENIZER,
        max_instances=None,
        lstm=True,
        max_seq_len=256,
        exclusive=False,
    )

    loader = MultiProcessDataLoader(
        reader,
        data_path=os.path.join(config.WIKI_DIR, "wiki.valid.tokens"),
        batch_size=2,
    )
    vocab = Vocabulary.from_files(config.VOCAB_DIR, padding_token="[PAD]", oov_token="[UNK]")
    loader.index_with(vocab)
    print("Ready...")
    for i in tqdm(loader):
        print(i["tokens"].size())
        # input()

    # Valid ratio: 1.1494
    # Test ratio:  1.1537
    # Train ratio: 1.1633

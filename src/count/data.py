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
from allennlp.data.fields import Field, TextField, MetadataField, TensorField
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


@DatasetReader.register("wikitext-reader")
class WikiTextReader(DatasetReader):
    def __init__(
        self,
        sequence_length: int,
        tokenizer_path: str,
        token_indexers: Dict[str, TokenIndexer] = None,
        exclusive: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.token_indexers = token_indexers
        self.exclusive = exclusive

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

        yield from self.tensor_to_instances(dataset["subwords"])

    def tensor_to_instances(self, subwords: torch.Tensor):
        num_sequences = (subwords.size(0) // self.sequence_length) * self.sequence_length
        sequences = subwords.narrow(0, 0, num_sequences).view(-1, self.sequence_length)
        logger.info("Yielding...")
        for inst in sequences:
            yield Instance({"tokens": TensorField(inst)})

    def text_to_instance(
        self,
        tokens: List[str],
        num_words: int,
    ) -> Instance:
        tokens = [Token(t) for t in tokens]
        return Instance({"tokens": TextField(tokens), "num_words": MetadataField(num_words)})

    def apply_token_indexers(self, instance) -> None:
        """Adds a token indexer to the instance. Automatically called by AllenNLP."""
        # instance["tokens"].token_indexers = self.token_indexers
        pass


if __name__ == "__main__":
    reader = WikiTextReader(
        sequence_length=128, tokenizer_path=config.TOKENIZER, max_instances=None
    )

    loader = MultiProcessDataLoader(
        reader, data_path=os.path.join(config.WIKI_DIR, "wiki.test.tokens"), batch_size=32
    )
    vocab = Vocabulary.from_files(config.VOCAB_DIR, padding_token="[PAD]", oov_token="[UNK]")
    loader.index_with(vocab)
    print("Ready...")
    for i in tqdm(loader):
        pass

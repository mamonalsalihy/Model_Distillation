import logging
import math
import os

# STL
import sys
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable
from tqdm import tqdm

# AllenNLP
from allennlp.data import Vocabulary
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

# Data types
from allennlp.data.fields import Field, TextField
from allennlp.data.instance import Instance

# Indexers
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer

# Tokenizers
from allennlp.data.tokenizers import Token, WhitespaceTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.tokenizers.tokenizer import Tokenizer

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Local
from src.count import config
from src.count.tokenizer import WikiTextTokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register("wikitext-reader", exist_ok=True)
class WikiTextReader(DatasetReader):
    """
    Creates `Instances` suitable for use in predicting a single next token using a language
    model. The :class:`Field`s that we create are the following:
    1. an input `TextField`
    2. target token `TextField`

    Parameters
    ----------
    tokenizer : `Tokenizer` (default=`WhitespaceTokenizer()`)
        We use this `Tokenizer` for the text.  See :class:`Tokenizer`.
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text, and to get ids for the mask
        targets.  See :class:`TokenIndexer`.
    max_tokens : `int`, optional (default = `None`)
        If you don't handle truncation at the `tokenizer` level, you can specify `max_tokens`
        here, and the only the last `max_tokens` will be used.
    split_on : `str`, optional (default = `"sentence"`)
        Determines the text to provide to the model.
        - `sentence` will split on sentences and provide one at a time
        - `paragraph` will provide sentences up until a newline.
    exclusive : `bool`, optional (default = `True`)
        If True, each generated instance will have no overlap with the previous. If False, each
        generated instance will have an overlap of all but 1 (i.e., shifted forward one token).
    min_context_len : `int`, optional (default = `None`)
        Determines the minimum number of tokens an instance must have in its context (not including
        the next token added as a target). If `None`, defaults to `context`. Otherwise, must be at
        least 1.
    """

    def __init__(
        self,
        context: int,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        split_on: str = "sentence",
        exclusive: bool = True,
        min_context_len: int = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._sentence_splitter = SpacySentenceSplitter(rule_based=True)
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._targets_tokenizer = self._tokenizer
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer(namespace="tokens")
        }
        self._split_on = split_on
        self._context = context
        self._exclusive = exclusive
        if min_context_len is None:
            self._min_context_len = self._context
        else:
            self._min_context_len = min_context_len
        # make sure we actually have a context
        assert self._min_context_len >= 1, f"min_context_len must be >= 1"

    def _read(self, file_path: str) -> Iterable[Instance]:
        logger.info(f"Loading data from {file_path}")
        with open(file_path, "r", encoding="utf8") as f:
            for line in self.shard_iterable(f):
                if line.strip() and line.strip()[0] != "=":
                    yield from self.generate_instances(line)

    def generate_instances(self, text: str) -> Iterable[Instance]:
        """Generates instances of a certain context size given the available text

        Arguments
        ---------
        text : str
            Text to tokenize and construct `self._context` sized instances out of

        Yields
        -------
        Iterable[Instance] :
            Generates `self._context` sized instances, where the `target` field is the next word
        """
        # tokenize the text, and slide a `self._context` sized window over the tokens , using the
        # (n+1)th token as a target.
        if self._split_on.lower() == "sentence":
            sentences = self._sentence_splitter.split_sentences(text)
            tokenized_sents = self._tokenizer.batch_tokenize(sentences)
            for tokens in tokenized_sents:
                # if we want exclusive instances, make sure the step size is self._context
                # otherwise, just use the normal 1
                step_size = self._context if self._exclusive else 1
                for start in range(0, len(tokens), step_size):
                    instance = tokens[start : start + self._context + 1]
                    # check whether the context length meets the minimum requirements
                    # don't forget to add 1 since the last token is not part of the context
                    if len(instance) >= self._min_context_len + 1:
                        yield self.text_to_instance(instance)
        elif self._split_on.lower() == "paragraph":
            # list of tokens in the paragraph
            tokens = []
            sentences = self._sentence_splitter.split_sentences(text)
            tokens = self._tokenizer.tokenize_paragraph(sentences, add_special_tokens=True)
            step_size = self._context if self._exclusive else 1
            for start in range(0, len(tokens), step_size):
                instance = tokens[start : start + self._context + 1]
                # check whether the context length meets the minimum requirements
                # don't forget to add 1 since the last token is not part of the context
                if len(instance) >= self._min_context_len + 1:
                    yield self.text_to_instance(instance)

        else:
            raise NotImplementedError(
                f"Splitting method {self._split_on} not implemented."
                " Please choose one of 'sentence' or 'paragraph'."
            )

    def text_to_instance(
        self,
        tokens: Iterable[Token],
    ) -> Instance:
        """Converts a list of `Token`s into an `Instance`

        Arguments
        ---------
        tokens : Iterable[Token]
            List of tokens to make into an instance. The last token in the list is the target,
            and the first `n-1` are the context.
        Returns
        -------
        Instance :
            Instance containing a `tokens` field and a `target` field.
        """

        tokens = TextField(tokens)
        fields: Dict[str, Field] = {"tokens": tokens}
        return Instance(fields)

    def apply_token_indexers(self, instance):
        """Adds a token indexer to the instance. Automatically called by AllenNLP."""
        instance["tokens"].token_indexers = self._token_indexers


if __name__ == "__main__":
    wiki_tokenizer = WikiTextTokenizer(
        tokenizer_path=config.TOKENIZER,
        add_special_tokens=True,
    )
    reader = WikiTextReader(
        context=256,
        tokenizer=wiki_tokenizer,
        token_indexers={"tokens": SingleIdTokenIndexer(namespace="tokens")},
        exclusive=True,
        split_on="sentence",
        max_instances=50_000,
        min_context_len=1,
        manual_distributed_sharding=True,
        manual_multiprocess_sharding=True,
    )
    # loader = MultiProcessDataLoader(
    #     reader=reader,
    #     data_path=os.path.join(config.WIKI_RAW_DIR, "wiki.train.raw"),
    #     batch_size=1,
    #     shuffle=True,
    #     max_instances_in_memory=None,
    #     num_workers=4,
    #     # start_method="spawn",
    # )
    # loader.index_with(vocab)
    dataset = reader.read(os.path.join(config.WIKI_RAW_DIR, "wiki.train.raw"))

    lens = [len(i.fields["tokens"]) for i in tqdm(dataset)]

    print(f"Max: {max(lens)}")
    print(f"Avg: {sum(lens) / len(lens)}")

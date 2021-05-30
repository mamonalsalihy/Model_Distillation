import logging
import math
import os

# STL
import sys
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, List
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
    context : `int`
        Maximum length of the context to use in prediction.
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
        - `paragraph` will provide a paragraph at a time.
        - `paragraph-with-seps` will provide a paragraph with each sentence having a `[SEP]` token
          in between
    exclusive : `bool`, optional (default = `True`)
        If True, each generated instance will have no overlap with the previous. If False, each
        generated instance will have an overlap of all but 1 (i.e., shifted forward one token).
    min_context_len : `int`, optional (default = `None`)
        Determines the minimum number of tokens an instance must have in its context (not including
        the next token added as a target). If `None`, defaults to `context`. Otherwise, must be at
        least 1.
    eval : `bool`, optional (default = `False`)
        If true, full context will be provided at each instance. That is, for each token in a
        sequence, the previous `context` tokens will be sent through.
    """

    def __init__(
        self,
        context: int,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        split_on: str = "sentence",
        exclusive: bool = True,
        min_context_len: int = None,
        eval: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._sentence_splitter = SpacySentenceSplitter(rule_based=True)
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._targets_tokenizer = self._tokenizer
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer(namespace="tokens")
        }

        # context stuff
        self._split_on = split_on
        self._context = context
        self._exclusive = exclusive
        if min_context_len is None:
            self._min_context_len = self._context
        else:
            self._min_context_len = min_context_len
        # make sure we actually have a context
        assert self._min_context_len >= 1, f"min_context_len must be >= 1"

        # check for eval mode
        self._eval = eval
        if self._eval:
            assert not self._exclusive, "Exclusive must be False for eval mode"

    def _read(self, file_path: str) -> Iterable[Instance]:
        logger.info(f"Loading data from {file_path}")
        with open(file_path, "r", encoding="utf8") as f:
            for line in self.shard_iterable(f):
                if line.strip() and line.strip()[0] != "=":
                    yield from self.generate_instances(line)

    def _batchify_tokens(self, tokens: List[Token]) -> Iterable[Instance]:
        # List of tokens -->
        step_size = self._context if self._exclusive else 1
        first_end_index = min(step_size, len(tokens) - 1)
        for end in range(first_end_index, len(tokens), step_size):
            start = max(0, end - self._context)
            instance = tokens[start : end + 1]
            if len(instance) >= self._min_context_len + 1:
                yield self.text_to_instance(instance)

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

        # 1. Optionally split text into sentences
        # 2. Tokenize & add cls/sep
        # 3. Yield instances of size `step_size` (1 if we are doing eval, context_len otherwise)

        step_size = self._context if self._exclusive else 1

        if self._split_on.lower() == "sentence":
            # Split on sentences with CLS and SEP at beg/end
            sentences = self._sentence_splitter.split_sentences(text)
            tokenized_sents = self._tokenizer.batch_tokenize(sentences)
            for tokens in tokenized_sents:
                yield from self._batchify_tokens(tokens)
        elif self._split_on.lower() == "paragraph":
            # Split on paragraphs with CLS and SEP at beg/end
            tokens = self._tokenizer.tokenize(text)
            yield from self._batchify_tokens(tokens)
        elif self._split_on.lower() == "paragraph-with-seps":
            # Split on paragraphs with SEP tokens between sentences
            sentences = self._sentence_splitter.split_sentences(text)
            tokens = self._tokenizer.tokenize_paragraph(sentences, add_special_tokens=True)
            yield from self._batchify_tokens(tokens)
        else:
            raise NotImplementedError(f"Splitting method {self._split_on} not implemented.")

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
        exclusive=False,
        split_on="paragraph-with-seps",
        eval=True,
        max_instances=256,
        min_context_len=1,
        manual_distributed_sharding=True,
        manual_multiprocess_sharding=True,
    )
    dataset = reader.read(os.path.join(config.WIKI_RAW_DIR, "wiki.train.raw"))

    print("Ready...")
    for i in dataset:
        print(i.fields["tokens"])

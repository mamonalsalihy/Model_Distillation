import logging
from itertools import islice
from typing import Dict, Iterable

# AllenNLP
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

# Local
from tokenizer import WikiTextTokenizer
import config

logger = logging.getLogger(__name__)


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
    """

    def __init__(
        self,
        context: int,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._sentence_splitter = SpacySentenceSplitter(rule_based=True)
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._targets_tokenizer = self._tokenizer
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer(namespace="tokens")
        }
        self._context = context

    def _read(self, file_path: str) -> Iterable[Instance]:
        logger.info(f"Loading data from {file_path}")
        with open(file_path, "r",encoding='utf8') as f:
            for line in f:
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
        # tokenize the text, and slide a `self._context` sized window over the tokens
        # , using the (n+1)th token as a target.
        sentences = self._sentence_splitter.split_sentences(text)
        for sent in sentences:
            tokens = self._tokenizer.tokenize(sent)
            for start in range(len(tokens) - self._context):
                width = start + self._context
                yield self.text_to_instance(tokens[start : width + 1])

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

        input_field = TextField(tokens, self._token_indexers)
        fields: Dict[str, Field] = {"tokens": input_field}
        return Instance(fields)


if __name__ == "__main__":
    wiki_tokenizer = WikiTextTokenizer(
        tokenizer_path=config.TOKENIZER,
        add_special_tokens=True,
    )
    reader = WikiTextReader(context=10, tokenizer=wiki_tokenizer)
    dataset = reader.read(config.WIKI_RAW_DIR / "wiki.train.raw")
    for i in islice(dataset, 4):
        print(i)

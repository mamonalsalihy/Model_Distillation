import copy
import logging
from typing import Dict, Iterable, List, cast

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import Token, WhitespaceTokenizer
from allennlp.data.tokenizers.tokenizer import Tokenizer

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
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._targets_tokenizer = self._tokenizer
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer(namespace="tokens")
        }
        self._context = context

    def _read(self, file_path: str) -> Iterable[Instance]:
        logger.info(f"Loading data from {file_path}")
        with open(file_path, "r") as f:
            for line in f:
                if line.strip() and line.strip()[0] != "=":
                    yield from self.generate_instances(line.replace("\n", "<eos>"))

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
        tokens = self._tokenizer.tokenize(text)
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

        input_field = TextField(tokens[:-1], self._token_indexers)
        target_field = TextField(tokens[-1:], self._token_indexers)
        fields: Dict[str, Field] = {"tokens": input_field, "target": target_field}
        return Instance(fields)


if __name__ == "__main__":
    reader = WikiTextReader(100)
    dataset = list(reader.read("../wikitext-103/wiki.mini.tokens"))

# AllenNLP
from allennlp.data import Vocabulary

# Local
import sys

from ..tokenizer import WikiTextTokenizer


if __name__ == "__main__":
    wiki_tokenizer = WikiTextTokenizer(
        tokenizer_path="../data/wikitext-tokenizer.json",
        add_special_tokens=True,
    )
    # Load the tokenizer vocabulary
    mapping = wiki_tokenizer.tokenizer.get_vocab()

    # Sort the indices so the indices of the tokenizer matches those of the vocabulary
    tokens, indices = zip(*sorted(mapping.items(), key=lambda x: x[1]))

    # Build the vocabulary from the tokenizer tokens
    vocab = Vocabulary(tokens_to_add={"tokens": tokens}, padding_token="[PAD]", oov_token="[UNK]")

    # Make sure the indices match up
    assert all(
        [
            wiki_tokenizer.tokenizer.id_to_token(i) == vocab.get_token_from_index(i)
            for i in range(32000)
        ]
    )

    # Save the vocabulary to disk
    vocab.save_to_files("../data/vocab/")

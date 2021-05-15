# AllenNLP
from allennlp.data import Vocabulary

# Local
import sys

sys.path.append("../")
from count import tokenizer, config


if __name__ == "__main__":
    wiki_tokenizer = tokenizer.WikiTextTokenizer(
        tokenizer_path=config.TOKENIZER,
        add_special_tokens=True,
    )
    # Load the tokenizer vocabulary
    mapping = wiki_tokenizer.tokenizer.get_vocab()

    # Sort the indices so the indices of the tokenizer matches those of the vocabulary
    tokens, indices = zip(*sorted(mapping.items(), key=lambda x: x[1]))

    # Build the vocabulary from the tokenizer tokens
    vocab = Vocabulary(
        tokens_to_add={"tokens": tokens}, padding_token=config.PAD, oov_token=config.UNK
    )

    # Make sure the indices match up
    assert all(
        [
            wiki_tokenizer.tokenizer.id_to_token(i) == vocab.get_token_from_index(i)
            for i in range(config.VOCAB_SIZE)
        ]
    )

    # Save the vocabulary to disk
    vocab.save_to_files(config.VOCAB_DIR)

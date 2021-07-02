from collections import Counter
from tqdm import tqdm

# AllenNLP
from allennlp.data import Vocabulary

# Local
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.count import tokenizer, config
from src.count.data import WikiTextReader


def from_tokenizer():
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
    return vocab


def from_file():
    counter = Counter()
    with open("../../data/wikitext-103/wiki.train.tokens", "r") as f:
        for line in tqdm(f):
            line = line.replace("<unk>", "[UNK]")
            line = line.replace("\n", "[SEP]")
            counter.update(line.split())
    vocab = Vocabulary({"tokens": counter}, padding_token=config.PAD, oov_token=config.UNK)
    vocab.save_to_files("../../data/word-level-vocab/")
    return vocab


if __name__ == "__main__":
    vocab = from_file()
    # vocab = Vocabulary.from_files(
    #     "../../data/word-level-vocab",
    #     padding_token=config.PAD,
    #     oov_token=config.UNK,
    # )
    print("Vocab size: ", vocab.get_vocab_size())

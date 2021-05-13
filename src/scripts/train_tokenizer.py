from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordPiece, WordLevel
from tokenizers.pre_tokenizers import Whitespace, Sequence, Digits
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer, WordLevelTrainer


def train(
    algorithm: str = "bpe",
    files: list = ["../../data/wikitext-103/wiki.train.raw"],
    output: str = "../../data/wikitext-tokenizer.json",
    vocab_size: int = 32_000,
    pre_tokenizers: list = None,
):

    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    # Initialize the classes
    # ======================
    if algorithm.lower() == "bpe":
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
    elif algorithm.lower() == "unigram":
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(
            unk_token="[UNK]", special_tokens=special_tokens, vocab_size=vocab_size
        )
    elif algorithm.lower() == "wordpiece":
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        trainer = WordPieceTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
    elif algorithm.lower() == "wordlevel":
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        trainer = WordLevelTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
    else:
        raise NotImplementedError(f"Method {algorithm} has not been added yet")

    # Collect the pretokenizers
    # =========================
    if pre_tokenizers is None:
        pre_tokenizers = Whitespace()
    else:
        pre_tokenizers = Sequence(pre_tokenizers)

    # Train and save tokenizer
    # ========================
    tokenizer.train(files, trainer)
    tokenizer.save(output)


if __name__ == "__main__":
    ALGORITHM = "unigram"
    FILES = ["../../data/wikitext-103/wiki.train.raw"]
    OUTPUT = "../../data/unigram-tokenizer.json"
    VOCAB_SIZE = 8_000
    PRE_TOKENIZERS = [Whitespace(), Digits(individual_digits=False)]

    train(
        algorithm=ALGORITHM,
        files=FILES,
        output=OUTPUT,
        vocab_size=VOCAB_SIZE,
        pre_tokenizers=PRE_TOKENIZERS,
    )

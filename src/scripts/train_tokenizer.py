import os

from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordPiece, WordLevel
from tokenizers import pre_tokenizers
from tokenizers import normalizers
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer, WordLevelTrainer

# Local
import sys;

sys.path.append('../')
from count import tokenizer, config


def train(
        algorithm: str = "bpe",
        files: list = [str(config.WIKI_RAW_DIR / "wiki.train.raw")],
        output: str = config.TOKENIZER,
        vocab_size: int = 32_000,
        pre: list = None,
        norms: list = None,
):
    special_tokens = [config.PAD, config.UNK, config.CLS, config.SEP]
    # Initialize the classes
    # ======================
    if algorithm.lower() == "bpe":
        tokenizer = Tokenizer(BPE(unk_token=config.UNK))
        trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
    elif algorithm.lower() == "unigram":
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(
            unk_token=config.UNK, special_tokens=special_tokens, vocab_size=vocab_size
        )
    elif algorithm.lower() == "wordpiece":
        tokenizer = Tokenizer(WordPiece(unk_token=config.UNK))
        trainer = WordPieceTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
    elif algorithm.lower() == "wordlevel":
        tokenizer = Tokenizer(WordLevel(unk_token=config.UNK))
        trainer = WordLevelTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
    else:
        raise NotImplementedError(f"Method {algorithm} has not been added yet")

    # Collect the pretokenizers
    # =========================
    if pre_tokenizers is None:
        pre = pre_tokenizers.Whitespace()
    else:
        pre = pre_tokenizers.Sequence(pre)

    if norms is None:
        tokenizer.normalizer = normalizers.NFKC()
    else:
        tokenizer.normalizer = normalizers.Sequence(norms)
    tokenizer.pre_tokenizer = pre

    # Train and save tokenizer
    # ========================
    tokenizer.train(files, trainer)
    tokenizer.save(output)


if __name__ == "__main__":
    ALGORITHM = "unigram"
    print(os.path.join(config.WIKI_RAW_DIR, "wiki.train.raw"))
    FILES = [os.path.join((config.WIKI_RAW_DIR / "wiki.train.raw"))]
    OUTPUT = config.TOKENIZER
    VOCAB_SIZE = 30_000
    PRE_TOKENIZERS = [pre_tokenizers.Whitespace(), pre_tokenizers.ByteLevel()]

    train(
        algorithm=ALGORITHM,
        files=FILES,
        output=OUTPUT,
        vocab_size=VOCAB_SIZE,
        pre=PRE_TOKENIZERS,
    )

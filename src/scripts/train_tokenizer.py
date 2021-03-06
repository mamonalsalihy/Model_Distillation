from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordPiece, WordLevel
from tokenizers import pre_tokenizers
from tokenizers import processors
from tokenizers import normalizers
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer, WordLevelTrainer
from tokenizers import decoders

# Local
import sys

sys.path.append("../")
from src.count import config
import os


def train(
    algorithm: str = "bpe",
    files: list = [os.path.join(config.WIKI_RAW_DIR, "wiki.train.raw")],
    output: str = config.TOKENIZER,
    vocab_size: int = 32_000,
    pre: list = None,
    norms: list = None,
    post: str = None,
):
    special_tokens = [config.PAD, config.UNK, config.CLS, config.SEP]
    # Initialize the classes
    # ======================
    if algorithm.lower() == "bpe":
        tokenizer = Tokenizer(BPE(unk_token=config.UNK))
        trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
        tokenizer.decoder = decoders.BPEDecoder()
    elif algorithm.lower() == "unigram":
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(
            unk_token=config.UNK, special_tokens=special_tokens, vocab_size=vocab_size
        )
    elif algorithm.lower() == "wordpiece":
        tokenizer = Tokenizer(WordPiece(unk_token=config.UNK))
        trainer = WordPieceTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
        tokenizer.decoder = decoders.WordPiece()
    elif algorithm.lower() == "wordlevel":
        tokenizer = Tokenizer(WordLevel(unk_token=config.UNK))
        trainer = WordLevelTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
    else:
        raise NotImplementedError(f"Method {algorithm} has not been added yet")

    # Add all the processors
    # =========================
    if pre_tokenizers is None:
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    else:
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pre)

    if norms is None:
        tokenizer.normalizer = normalizers.NFKC()
    else:
        tokenizer.normalizer = normalizers.Sequence(norms)

    # Train and save tokenizer
    # ========================
    print("Training tokenizer ... ")
    tokenizer.train(files, trainer)

    if post.lower() == "bert":
        sep_idx = tokenizer.token_to_id(config.SEP)
        cls_idx = tokenizer.token_to_id(config.CLS)
        tokenizer.post_processor = processors.BertProcessing(
            sep=(config.SEP, sep_idx), cls=(config.CLS, cls_idx)
        )

    tokenizer.save(output)
    print("Finished training tokenizer ... ")


if __name__ == "__main__":
    ALGORITHM = "wordpiece"
    FILES = [os.path.join(config.WIKI_RAW_DIR, "wiki.train.raw")]
    OUTPUT = config.TOKENIZER
    VOCAB_SIZE = 30_000
    PRE_TOKENIZERS = [pre_tokenizers.Whitespace(), pre_tokenizers.ByteLevel()]
    FILES = [os.path.join(config.WIKI_RAW_DIR, "wiki.train.raw")]
    OUTPUT = config.TOKENIZER
    VOCAB_SIZE = 32_000
    PRE_TOKENIZERS = [
        pre_tokenizers.Whitespace(),
        pre_tokenizers.BertPreTokenizer(),
        pre_tokenizers.Digits(individual_digits=False),
    ]
    NORMS = [normalizers.BertNormalizer(lowercase=False)]
    POST_PROCESSOR = "BERT"

    train(
        algorithm=ALGORITHM,
        files=FILES,
        output=OUTPUT,
        vocab_size=VOCAB_SIZE,
        pre=PRE_TOKENIZERS,
        norms=NORMS,
        post=POST_PROCESSOR,
    )

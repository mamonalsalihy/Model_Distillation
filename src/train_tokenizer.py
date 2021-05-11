from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

if __name__ == "__main__":
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"], vocab_size=32000)

    tokenizer.pre_tokenizer = Whitespace()

    # ignore validation and test data
    files = [f"../data/wikitext-103/wiki.{split}.raw" for split in ["train"]]

    # train on just the train data
    tokenizer.train(files, trainer)

    tokenizer.save("../data/wikitext-tokenizer.json")

    # example usage
    output = tokenizer.encode("Hello this is going to be tokenized")

    print(output.tokens)

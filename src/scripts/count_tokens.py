from tokenizers import Tokenizer
from tqdm import tqdm

if __name__ == "__main__":
    tokenizer = Tokenizer.from_file("../../wordpiece-tokenizer.json")

    total_tokenized = 0
    total_raw = 0
    with open("../../data/wikitext-103-raw/wiki.train.raw", "r") as f:
        for line in tqdm(f):
            if line.strip() and line.strip()[0] != "=":
                # tokens = tokenizer.encode(line.strip())
                # total_tokenized += len(tokens)
                total_raw += len(line.split())
    # print(total_tokenized)
    print(total_raw)


# 114,661,592
# 99,183,639

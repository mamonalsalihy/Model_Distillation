from itertools import chain

import pandas as pd
import os

import sys

from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter

sys.path.append('../count')

from tokenizer import WikiTextTokenizer
import config


def main():
    # This scripts takes very long to execute due to processing 100+ million tokens
    print("Getting dataset statistics ... ")
    wiki_tokenizer = WikiTextTokenizer(
        tokenizer_path=config.TOKENIZER,
        add_special_tokens=True,
    )
    sentence_splitter = SpacySentenceSplitter(rule_based=True)
    fd = open(os.path.join(config.WIKI_RAW_DIR, "wiki.train.raw"), encoding='utf8')
    sentences = [sentence_splitter.split_sentences(line) for line in fd.read() if line.strip() and line.strip()[0] != '=']
    tokens = [wiki_tokenizer.tokenize(sentence) for sentence in sentences]
    tokens_ = list(chain.from_iterable(tokens))

    print("Average sentence length: {}".format(sum(len(sentence) for sentence in sentences) / len(sentences)))
    print("Total number of tokens: {}".format(len(tokens_)))




if __name__ == '__main__':
    main()

# STL
import sys
from pathlib import Path
import argparse

# Torch
import torch

# AllenNLP
from allennlp.models import Model
from allennlp.common import Params
from allennlp.data.fields import Field, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

# Local
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Local
from src.count import config
from src.count.tokenizer import WikiTextTokenizer
from src.count.decoders.transformer_decoder import TransformerDecoder
from src.count.models.simple_transformer import SimpleTransformerLanguageModel


class LMInference:
    def __init__(self, model: Model, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.indexer = {"tokens": SingleIdTokenIndexer(namespace="tokens")}

        # only for evaluation
        self.model.eval()

    def _make_instance(self, tokens):
        instance = Instance({"tokens": TextField(tokens)})
        instance["tokens"].token_indexers = self.indexer
        return instance

    def predict_continuation(self, text: str, n: int):
        new_text = text
        for i in range(n):
            tokens = self.tokenizer.tokenize(new_text)
            ids = self.indexer["tokens"].tokens_to_indices(tokens, self.model.vocab)["tokens"]
            instance = self._make_instance(tokens)
            with torch.no_grad():
                output = self.model.forward_on_instance(instance)
            new_ids = output["token_ids"]
            ids.append(int(new_ids))
            new_text = self.tokenizer.tokenizer.decode(ids)

        return new_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer")
    parser.add_argument("archive_dir")
    args = parser.parse_args()

    tokenizer = WikiTextTokenizer(
        tokenizer_path=args.tokenizer,
        add_special_tokens=True,
    )
    params = Params.from_file(Path(args.archive_dir) / "config.json")
    model = Model.load(params, serialization_dir=args.archive_dir)
    inf = LMInference(model, tokenizer)

    print(inf.predict_continuation("In 1867, Andrew Jackson fought", 10))

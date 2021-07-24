# STL
import sys
from pathlib import Path
import argparse

# Torch
import torch

# AllenNLP
from allennlp.models import Model
from allennlp.common import Params
from allennlp.data.fields import Field, TextField, TensorField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from tokenizers import Tokenizer

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

        # only for evaluation
        self.model.eval()

    def predict_continuation(self, text: str, n: int, ratio: float):
        ids = self.tokenizer.encode(text).ids[:-1]
        for i in range(n):
            x = torch.tensor(ids, dtype=torch.long, device="cpu").view(1, -1)
            with torch.no_grad():
                output = self.model.forward(x, ratio, only_predict_next=True)
            output = self.model.make_output_human_readable(output)
            new_ids = list(output["token_ids"])
            ids.append(new_ids[-1])
        new_text = self.tokenizer.decode(ids)

        return new_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer")
    parser.add_argument("archive_dir")
    parser.add_argument("max")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    params = Params.from_file(Path(args.archive_dir) / "config.json")
    model = Model.load(params, serialization_dir=args.archive_dir)
    inf = LMInference(model, tokenizer)

    print(inf.predict_continuation("Super Mario Bros. is a platform game", int(args.max)))
    # print(tokenizer.decode([2,  8562, 26606,  6591,  6617]))

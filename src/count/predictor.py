# STL
import sys
from pathlib import Path
import argparse

# Torch
import torch
from collections import namedtuple

# AllenNLP
from allennlp.models import Model
from allennlp.common import Params
from allennlp.data.tokenizers.tokenizer import Tokenizer

from tokenizers import Tokenizer

# Local
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.count.models.simple_transformer import SimpleTransformerLanguageModel
from src.count.decoders.transformer_decoder import TransformerDecoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LMInference:
    def __init__(self, model: Model, tokenizer: Tokenizer, backwards: bool = False):
        self.model = model.to(DEVICE)
        self.tokenizer = tokenizer

        # only for evaluation
        self.model.eval()
        self.backwards = backwards

    def speak(self, text: str, n: int, temperature: float):
        if self.backwards:
            self.speak_backwards(text=text, n=n, temperature=temperature)
        else:
            self.speak_forwards(text=text, n=n, temperature=temperature)

    def speak_forwards(self, text: str, n: int, temperature: float):
        for i in range(n):
            ids = self.tokenizer.encode(text).ids
            x = torch.tensor(ids, dtype=torch.long, device=DEVICE).view(1, -1)
            with torch.no_grad():
                output = self.model.forward(x, 1.0)
            # output = self.model.make_output_human_readable(output)
            #         backward = torch.flip(backward, dims=[0])
            logits = torch.flip(output['logits'], dims=[0])
            logits = logits.view(1, -1, logits.size()[-1])
            tokens = torch.argmax(logits / temperature, dim=-1)
            new_id = tokens[:, -1].item()
            ids.append(new_id)
            text = self.tokenizer.decode(ids)
            if new_id == self.tokenizer.token_to_id("[CLS]"):
                return text
        return self.tokenizer.decode(ids)

    def speak_backwards(self, text: str, n: int, temperature: float):
        for i in range(n):
            ids = self.tokenizer.encode(text).ids
            x = torch.tensor(ids, dtype=torch.long, device=DEVICE).view(1, -1)
            with torch.no_grad():
                output = self.model.forward(x, 1.0)
            # output = self.model.make_output_human_readable(output)
            logits = output['logits']
            logits = logits.view(1, -1, logits.size()[-1])
            tokens = torch.argmax(logits / temperature, dim=-1)
            new_id = tokens[:, 0].item()
            ids.insert(0, new_id)
            text = self.tokenizer.decode(ids)
            if new_id == self.tokenizer.token_to_id("[CLS]"):
                return text
        return self.tokenizer.decode(ids)


def load(args=None):
    if args is None:
        args = {"tokenizer": str(Path(__file__).parents[2].resolve() / "wordpiece-tokenizer.json"),
                "archive_dir": str(Path(__file__).parents[2].resolve() / "saved-experiments/138M-model/")}
        print(args)
        args = namedtuple("args", args)(**args)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    params = Params.from_file(Path(args.archive_dir) / "config.json")
    model = Model.load(params, serialization_dir=args.archive_dir)
    inf = LMInference(model, tokenizer)
    return inf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer")
    parser.add_argument("archive_dir")
    parser.add_argument("max")
    parser.add_argument("text")
    args = parser.parse_args()

    inf = load(args)
    print(inf.speak(args.text, int(args.max)))

# AllenNLP
# Local
import sys
from pathlib import Path

# Torch
import torch.nn as nn
from allennlp.common.registrable import Registrable

sys.path.append(str(Path(__file__).resolve().parents[3]))

class Decoder(nn.Module, Registrable):
    def __init__(self, **kwargs):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

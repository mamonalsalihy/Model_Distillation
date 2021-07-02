from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

# Local
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

# Local
from src.count.decomposed_layers.linear import KLinear


class KEmbedding(nn.Module):
    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ]

    num_embeddings: int
    embedding_dim: int
    padding_idx: Optional[int]
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    sparse: bool

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        rank: int,
        sizes: List[int],
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(KEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.klinear = KLinear(
            num_embeddings,
            embedding_dim,
            rank,
            sizes,
            bias=False,
            **factory_kwargs,
        )
        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq

        if _weight is None:
            self.weight = self.klinear.weight
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [
                num_embeddings,
                embedding_dim,
            ], "Shape of weight does not match num_embeddings and embedding_dim"
            self.weight = Parameter(_weight)

        self.sparse = sparse

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        return F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        if self.sparse is not False:
            s += ", sparse=True"
        return s.format(**self.__dict__)

    @classmethod
    def from_pretrained(
        cls,
        embeddings,
        freeze=True,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
    ):
        r"""Creates Embedding instance from given 2-dimensional FloatTensor."""
        assert embeddings.dim() == 2, "Embeddings parameter is expected to be 2-dimensional"
        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
        embedding.weight.requires_grad = not freeze
        return embedding


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    millions = total // 1_000_000
    thousands = (total - millions * 1_000_000) // 1_000
    string = str(millions) + "." + str(thousands) + "M"
    return string


if __name__ == "__main__":
    num_embeddings = 32_000
    embedding_dim = 768
    rank = 8

    f = lambda n: [(i, n // i) for i in range(2, int(n ** 0.5) + 1) if n % i == 0]
    input = reversed(f(num_embeddings))
    output = f(embedding_dim)

    sizes = []
    for (l1, r1), (l2, r2) in zip(input, output):
        sizes.append([(l1, l2), (r1, r2)])

    print(sum(l1 * l2 + r1 * r2 for (l1, l2), (r1, r2) in sizes) * rank)
    print(sizes)

    # a = ((4_000, 12), (8, 64))
    # b = ((1_000, 24), (32, 32))
    # c = ((500, 48), (64, 16))
    # d = ((250, 48), (128, 16))
    # e = ((125, 48), (256, 16))
    # sizes = [a, b, c, d, e]
    padding_idx = 0

    small = KEmbedding(num_embeddings, embedding_dim, rank, sizes, padding_idx)
    big = nn.Embedding(num_embeddings, embedding_dim)

    input = torch.tensor([1, 2, 3, 4, 5])
    embs = small(input)
    print("Embeddings: ", embs)
    print("Shapes: ", embs.shape)
    print("Small parameters: ", count_parameters(small))
    print("Big parameters: ", count_parameters(big))

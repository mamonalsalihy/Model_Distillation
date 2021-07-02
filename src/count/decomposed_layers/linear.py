# STL
from typing import Tuple, List

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F


class KLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        factor_sizes: List,
        bias=True,
        device=None,
        dtype=None,
    ) -> None:
        """Linear Layer which stores weights as a Kronecker product of factor matrices

        Arguments
        ---------
        in_features : int
            Number of features input vectors will have
        out_features : int
            Number of features output vectors will have
        rank : int
            Number of sets of factors to use. For example, if `factor_sizes` was [[(8, 8), (4, 12)],
            [(4, 12), (8, 8)]] and rank was 2, we would have 2 sets of 2 pairs of factors, giving us
            4 total pairs of parameters.
        factor_sizes : List[Tuple[size, size]], (size = Tuple[int, int])
            Left/right sizes of each factor pair to use. In the above example, we would have
            - Pair 1: 8x8 (left) and 4x12 (right)
            - Pair 2: 4x12 (left) and 8x8 (right)
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.shape = (in_features, out_features)
        self.rank = rank
        self.factor_sizes = factor_sizes

        left_factors = []
        right_factors = []
        for r in range(rank):
            for left_size, right_size in self.factor_sizes:
                assert left_size[0] * right_size[0] == in_features
                assert left_size[1] * right_size[1] == out_features
                left = nn.Parameter(torch.rand(size=left_size, **factory_kwargs))
                right = nn.Parameter(torch.rand(size=right_size, **factory_kwargs))
                left_factors.append(left)
                right_factors.append(right)

        self.left_factors = nn.ParameterList(left_factors)
        self.right_factors = nn.ParameterList(right_factors)

        self.update()
        if bias:
            self.bias = nn.Parameter(torch.rand(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def update(self):
        weight = torch.zeros(size=self.shape)
        for left, right in zip(self.left_factors, self.right_factors):
            weight += torch.kron(left, right)

        self.weight = weight.T

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight, self.bias)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        millions = total // 1_000_000
        thousands = (total - millions * 1_000_000) // 1_000
        string = str(millions) + "." + str(thousands) + "M"
        return string


def test():
    # Setup
    in_features = 512
    out_features = 2048
    a = ((8, 16), (64, 128))
    b = ((32, 256), (16, 8))
    c = ((16, 32), (32, 64))
    m = KLinear(in_features, out_features, rank=4, factor_sizes=[a, b, c], bias=False)
    optim = torch.optim.Adam(m.parameters(), lr=1e-1)

    # Compute
    x = torch.rand((4, in_features))
    y = m(x)
    target = torch.ones(4, dtype=torch.long)

    # Save prior weights
    before = m.weight

    # Calculate loss & update params
    loss = F.cross_entropy(y, target)
    loss.backward()
    optim.step()
    m.update()

    # Post weights
    after = m.weight

    assert not (before == after).all(), before - after

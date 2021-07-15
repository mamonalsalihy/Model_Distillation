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
        factor_sizes: List = None,
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
        self.factor_sizes = factor_sizes or self.from_factors()

        left_factors = []
        right_factors = []
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

    def _hook(self, module, grad_input, grad_output):
        self.update()

    def update(self):
        weight = torch.zeros(size=self.shape)
        for left, right in zip(self.left_factors, self.right_factors):
            weight += torch.kron(left, right)

        self.weight = weight.T / len(self.left_factors)

    def forward(self, x: torch.Tensor):
        self.update()
        return F.linear(x, self.weight, self.bias)

    def from_factors(self):
        in_size, out_size = self.shape
        in_sizes = [i for i in range(2, in_size) if in_size % i == 0]
        out_sizes = [i for i in range(2, out_size) if out_size % i == 0]

        firsts = zip(in_sizes, in_sizes[::-1])
        seconds = zip(out_sizes, out_sizes[::-1])

        lefts = []
        rights = []
        for (l1, r1), (l2, r2) in zip(firsts, seconds):
            lefts.append((l1, l2))
            rights.append((r1, r2))

        return list(zip(lefts, rights))

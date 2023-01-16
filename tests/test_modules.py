import random
from typing import List

from hypothesis import given
from hypothesis.strategies import DrawFn, composite, floats, integers, lists

import simpletorch
from simpletorch import Parameter, Scalar


@composite
def scalars(
    draw: DrawFn, min_value: float = -100000, max_value: float = 100000
) -> Scalar:
    val = draw(floats(min_value=min_value, max_value=max_value))
    return simpletorch.Scalar(val)


class Network(simpletorch.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = ScalarLinear(2, 1)


class Network2(simpletorch.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = ScalarLinear(2, 2)
        self.layer2 = ScalarLinear(2, 1)


class ScalarLinear(simpletorch.Module):
    def __init__(self, in_size: int, out_size: int) -> None:
        super().__init__()
        self.weights: List[List[Parameter]] = []
        self.bias: List[Parameter] = []
        for i in range(in_size):
            self.weights.append([])
            for j in range(out_size):
                self.weights[i].append(
                    self.add_parameter(
                        f"weight_{i}_{j}",
                        simpletorch.Scalar(2 * (random.random() - 0.5)),
                    )
                )
        for j in range(out_size):
            self.bias.append(
                self.add_parameter(
                    f"bias_{j}", simpletorch.Scalar(2 * (random.random() - 0.5))
                )
            )

    def forward(self, inputs: List[Scalar]) -> List[Scalar]:
        y = [b.value for b in self.bias]
        for i, x in enumerate(inputs):
            for j in range(len(y)):
                y[j] = y[j] + x * self.weights[i][j].value
        return y


@given(lists(scalars(), max_size=10), integers(min_value=5, max_value=20))
def test_linear(inputs: List[Scalar], out_size: int) -> None:
    lin = ScalarLinear(len(inputs), out_size)
    mid = lin.forward(inputs)
    lin2 = ScalarLinear(out_size, 1)
    lin2.forward(mid)


def test_nn_size() -> None:
    model = Network2()
    assert len(model.parameters()) == (
        len(model.layer1.parameters()) + len(model.layer2.parameters())
    )

    assert model.layer2.bias[0].value.data != 0
    assert model.layer1.bias[0].value.data != 0
    assert model.layer1.weights[0][0].value.data != 0

    for p in model.parameters():
        p.update(simpletorch.Scalar(0))

    assert model.layer2.bias[0].value.data == 0
    assert model.layer1.bias[0].value.data == 0
    assert model.layer1.weights[0][0].value.data == 0

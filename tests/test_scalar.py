from typing import Callable, Tuple

import pytest
from hypothesis import given
from hypothesis.strategies import DrawFn, composite, floats

import simpletorch
from simpletorch import (
    MathTestVariable,
    Scalar,
    central_difference,
    derivative_check,
    operators,
)

from .strategies import assert_close, small_floats


@composite
def scalars(
    draw: DrawFn, min_value: float = -100000, max_value: float = 100000
) -> Scalar:
    val = draw(floats(min_value=min_value, max_value=max_value))
    return simpletorch.Scalar(val)


small_scalars = scalars(min_value=-100, max_value=100)


def test_central_diff() -> None:
    d = central_difference(operators.id, 5, arg=0)
    assert_close(d, 1.0)

    d = central_difference(operators.add, 5, 10, arg=0)
    assert_close(d, 1.0)

    d = central_difference(operators.mul, 5, 10, arg=0)
    assert_close(d, 10.0)

    d = central_difference(operators.mul, 5, 10, arg=1)
    assert_close(d, 5.0)

    d = central_difference(operators.exp, 2, arg=0)
    assert_close(d, operators.exp(2.0))


@given(small_floats, small_floats)
def test_simple(a: float, b: float) -> None:
    # Simple add
    c = Scalar(a) + Scalar(b)
    assert_close(c.data, a + b)

    # Simple mul
    c = Scalar(a) * Scalar(b)
    assert_close(c.data, a * b)

    # Simple relu
    c = Scalar(a).relu() + Scalar(b).relu()
    assert_close(c.data, simpletorch.operators.relu(a) + simpletorch.operators.relu(b))


one_arg, two_arg, _ = MathTestVariable._comp_testing()


@given(small_scalars)
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(
    fn: Tuple[str, Callable[[float], float], Callable[[Scalar], Scalar]], t1: Scalar
) -> None:
    name, base_fn, scalar_fn = fn
    assert_close(scalar_fn(t1).data, base_fn(t1.data))


@given(small_scalars, small_scalars)
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(
    fn: Tuple[str, Callable[[float, float], float], Callable[[Scalar, Scalar], Scalar]],
    t1: Scalar,
    t2: Scalar,
) -> None:
    name, base_fn, scalar_fn = fn
    assert_close(scalar_fn(t1, t2).data, base_fn(t1.data, t2.data))


@given(small_scalars)
@pytest.mark.parametrize("fn", one_arg)
def test_one_derivative(
    fn: Tuple[str, Callable[[float], float], Callable[[Scalar], Scalar]], t1: Scalar
) -> None:
    name, _, scalar_fn = fn
    derivative_check(scalar_fn, t1)


@given(small_scalars, small_scalars)
@pytest.mark.parametrize("fn", two_arg)
def test_two_derivative(
    fn: Tuple[str, Callable[[float, float], float], Callable[[Scalar, Scalar], Scalar]],
    t1: Scalar,
    t2: Scalar,
) -> None:
    name, _, scalar_fn = fn
    derivative_check(scalar_fn, t1, t2)

import math
from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def sqrt(x: float) -> float:
    return math.sqrt(x)


def lt(x: float, y: float) -> float:
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    return math.isclose(x, y, abs_tol=1e-2)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    return math.log(x + EPS)


def exp(x: float) -> float:
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    return d / (2 * sqrt(x))


def inv(x: float) -> float:
    return 1 / x


def inv_back(x: float, d: float) -> float:
    return (-1 / x ** 2) * d


def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0.0


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    return lambda ls: [fn(item) for item in ls]


def negList(ls: Iterable[float]) -> Iterable[float]:
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    return lambda xls, yls: [fn(x, y) for x, y in zip(xls, yls)]


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:

    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    def f(ls):
        tmp = start
        for item in ls:
            tmp = fn(tmp, item)
        return tmp

    return f


def sum(ls: Iterable[float]) -> float:
    return reduce(add, 0)(ls)


def prod(ls: Iterable[float]) -> float:
    return reduce(mul, 1.0)(ls)

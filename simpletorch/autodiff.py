from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    vals_incre = list(vals)
    vals_deces = list(vals)
    vals_incre[arg] += epsilon
    vals_deces[arg] -= epsilon
    return (f(*vals_incre) - f(*vals_deces)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        return self.history.last_fn is None

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    order = []
    seen = set()

    def visit(var: Variable) -> None:
        if var.unique_id in seen:
            return
        if not var.is_leaf():
            for m in var.history.inputs:
                if not m.is_constant():
                    visit(m)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    var_queue = topological_sort(variable=variable)
    var_dict = {variable.unique_id: deriv}
    for var in var_queue:
        var_deriv = var_dict.get(var.unique_id)
        if var.is_leaf():
            var.accumulate_derivative(var_deriv)
        else:
            for back_var, back_deriv in var.chain_rule(var_deriv):
                curr_deriv = var_dict.get(back_var.unique_id, 0)
                var_dict[back_var.unique_id] = curr_deriv + back_deriv


@dataclass
class Context:
    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values

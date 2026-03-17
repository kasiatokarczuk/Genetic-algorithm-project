"""
Operator inwersji oraz funkcje testowe.
"""
import numpy as np
from .chromosome import Chromosome


# ── Operator inwersji ──────────────────────────────────────────────────────────

def inversion(c: Chromosome, prob: float = 0.01) -> Chromosome:
    """
    Operator inwersji: z prawdopodobieństwem prob odwraca losowo wybrany
    podciąg bitów chromosomu.
    """
    child = c.copy()
    if np.random.rand() < prob:
        pts = sorted(np.random.choice(c.total_bits + 1, size=2, replace=False))
        child.genes[pts[0]:pts[1]] = child.genes[pts[0]:pts[1]][::-1]
    return child


# ── Funkcje testowe ────────────────────────────────────────────────────────────

class RosenbrockFunction:
    """
    Funkcja Rosenbrocka (banan Rosenbrocka).

    f(x) = sum_{i=0}^{N-2} [ 100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2 ]

    Zakres:   [-2.048, 2.048]^N
    Minimum:  f(1, 1, ..., 1) = 0
    """
    name = "Rosenbrock"
    default_bounds = (-2.048, 2.048)
    global_minimum_value = 0.0

    def __init__(self, n_variables: int = 2):
        self.n_variables = n_variables
        self.bounds = [self.default_bounds] * n_variables
        self.global_minimum_x = [1.0] * n_variables

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(np.sum(
            100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2
        ))

    def __repr__(self):
        return f"RosenbrockFunction(n={self.n_variables})"


TEST_FUNCTIONS = {
    "rosenbrock": RosenbrockFunction,
}

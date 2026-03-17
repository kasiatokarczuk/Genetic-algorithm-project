"""
Mutacje: brzegowa, jednopunktowa, dwupunktowa.
"""
import numpy as np
from .chromosome import Chromosome


def mutation_edge(c: Chromosome, prob: float = 0.01) -> Chromosome:
    """
    Mutacja brzegowa: z prawdopodobieństwem prob zamienia pierwszy
    lub ostatni bit każdej zmiennej.
    """
    child = c.copy()
    for i in range(c.n_variables):
        if np.random.rand() < prob:
            # Losowo wybieramy bit brzegowy (pierwszy lub ostatni bit zmiennej)
            bit_idx = i * c.bits_per_var
            end_idx = bit_idx + c.bits_per_var - 1
            chosen = np.random.choice([bit_idx, end_idx])
            child.genes[chosen] ^= 1
    return child


def mutation_single_point(c: Chromosome, prob: float = 0.01) -> Chromosome:
    """Mutacja jednopunktowa: flip jednego losowego bitu."""
    child = c.copy()
    if np.random.rand() < prob:
        point = np.random.randint(c.total_bits)
        child.genes[point] ^= 1
    return child


def mutation_two_point(c: Chromosome, prob: float = 0.01) -> Chromosome:
    """Mutacja dwupunktowa: flip dwóch losowych bitów."""
    child = c.copy()
    if np.random.rand() < prob:
        pts = np.random.choice(c.total_bits, size=2, replace=False)
        child.genes[pts[0]] ^= 1
        child.genes[pts[1]] ^= 1
    return child


MUTATION_METHODS = {
    "edge": mutation_edge,
    "single_point": mutation_single_point,
    "two_point": mutation_two_point,
}

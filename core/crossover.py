"""
Krzyżowanie: jednopunktowe, dwupunktowe, jednorodne, ziarniste.
"""
import numpy as np
from .chromosome import Chromosome


def _make_child(parent1: Chromosome, genes: np.ndarray) -> Chromosome:
    return Chromosome.from_genes(genes, parent1.n_variables,
                                 parent1.bounds, parent1.precision)


def crossover_single_point(p1: Chromosome, p2: Chromosome) -> tuple[Chromosome, Chromosome]:
    """Krzyżowanie jednopunktowe."""
    point = np.random.randint(1, p1.total_bits)
    g1 = np.concatenate([p1.genes[:point], p2.genes[point:]])
    g2 = np.concatenate([p2.genes[:point], p1.genes[point:]])
    return _make_child(p1, g1), _make_child(p1, g2)


def crossover_two_point(p1: Chromosome, p2: Chromosome) -> tuple[Chromosome, Chromosome]:
    """Krzyżowanie dwupunktowe."""
    pts = sorted(np.random.choice(p1.total_bits - 1, size=2, replace=False) + 1)
    a, b = pts
    g1 = np.concatenate([p1.genes[:a], p2.genes[a:b], p1.genes[b:]])
    g2 = np.concatenate([p2.genes[:a], p1.genes[a:b], p2.genes[b:]])
    return _make_child(p1, g1), _make_child(p1, g2)


def crossover_uniform(p1: Chromosome, p2: Chromosome,
                      swap_prob: float = 0.5) -> tuple[Chromosome, Chromosome]:
    """Krzyżowanie jednorodne."""
    mask = np.random.rand(p1.total_bits) < swap_prob
    g1 = np.where(mask, p2.genes, p1.genes)
    g2 = np.where(mask, p1.genes, p2.genes)
    return _make_child(p1, g1), _make_child(p1, g2)


def crossover_granular(p1: Chromosome, p2: Chromosome,
                       grain_size: int = 4) -> tuple[Chromosome, Chromosome]:
    """Krzyżowanie ziarniste (wymiana bloków bitów)."""
    g1, g2 = p1.genes.copy(), p2.genes.copy()
    i = 0
    while i < p1.total_bits:
        end = min(i + grain_size, p1.total_bits)
        if np.random.rand() < 0.5:
            g1[i:end], g2[i:end] = p2.genes[i:end].copy(), p1.genes[i:end].copy()
        i += grain_size
    return _make_child(p1, g1), _make_child(p1, g2)


CROSSOVER_METHODS = {
    "single_point": crossover_single_point,
    "two_point": crossover_two_point,
    "uniform": crossover_uniform,
    "granular": crossover_granular,
}

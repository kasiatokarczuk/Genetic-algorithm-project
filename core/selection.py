"""
Metody selekcji: najlepszych, ruletki, turniejowa.
"""
import numpy as np
from .chromosome import Chromosome


def _fitnesses(population: list[Chromosome], maximize: bool) -> np.ndarray:
    raw = np.array([c.fitness for c in population], dtype=float)
    return raw if maximize else -raw


def selection_best(population: list[Chromosome], n: int,
                   maximize: bool = False) -> list[Chromosome]:
    """Selekcja n najlepszych osobników."""
    fit = _fitnesses(population, maximize)
    indices = np.argsort(fit)[::-1][:n]
    return [population[i].copy() for i in indices]


def selection_roulette(population: list[Chromosome], n: int,
                       maximize: bool = False) -> list[Chromosome]:
    """Selekcja metodą ruletki (proporcjonalna do przystosowania)."""
    fit = _fitnesses(population, maximize)
    # Przesunięcie aby wszystkie wartości były dodatnie
    fit = fit - fit.min() + 1e-9
    probs = fit / fit.sum()
    indices = np.random.choice(len(population), size=n, replace=True, p=probs)
    return [population[i].copy() for i in indices]


def selection_tournament(population: list[Chromosome], n: int,
                         maximize: bool = False,
                         tournament_size: int = 3) -> list[Chromosome]:
    """Selekcja turniejowa."""
    selected = []
    fit = _fitnesses(population, maximize)
    for _ in range(n):
        competitors = np.random.choice(len(population), size=tournament_size, replace=False)
        winner = competitors[np.argmax(fit[competitors])]
        selected.append(population[winner].copy())
    return selected


SELECTION_METHODS = {
    "best": selection_best,
    "roulette": selection_roulette,
    "tournament": selection_tournament,
}

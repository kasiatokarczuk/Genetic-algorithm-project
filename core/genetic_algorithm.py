"""
Główna pętla algorytmu genetycznego.
"""
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from .chromosome import Chromosome
from .selection import SELECTION_METHODS
from .crossover import CROSSOVER_METHODS
from .mutation import MUTATION_METHODS
from .operators import inversion


@dataclass
class GAConfig:
    # Parametry populacji
    population_size: int = 50
    n_epochs: int = 100
    n_variables: int = 2
    bounds: list = field(default_factory=lambda: [(-2.048, 2.048)])
    precision: int = 6

    # Operatory
    selection_method: str = "tournament"    # best | roulette | tournament
    crossover_method: str = "single_point"  # single_point | two_point | uniform | granular
    mutation_method: str = "single_point"   # edge | single_point | two_point
    crossover_prob: float = 0.8
    mutation_prob: float = 0.01
    inversion_prob: float = 0.01
    tournament_size: int = 3

    # Strategia elitarna
    elite_size: int = 1

    # Cel optymalizacji
    maximize: bool = False


@dataclass
class GAResult:
    best_individual: Chromosome
    best_fitness: float
    best_x: np.ndarray
    history_best: list[float]
    history_avg: list[float]
    elapsed_time: float
    n_epochs_run: int


class GeneticAlgorithm:
    """
    Klasyczny algorytm genetyczny z binarną reprezentacją chromosomu.
    """

    def __init__(self, func: Callable, config: GAConfig,
                 progress_callback: Callable = None):
        """
        func:              funkcja celu f(x: np.ndarray) -> float
        config:            obiekt GAConfig
        progress_callback: opcjonalna funkcja(epoch, best_fitness, avg_fitness)
                           wywoływana po każdej epoce (do aktualizacji GUI)
        """
        self.func = func
        self.cfg = config
        self.progress_callback = progress_callback
        self._stop_flag = False

    def stop(self):
        """Zatrzymaj algorytm po bieżącej epoce."""
        self._stop_flag = True

    # ── Inicjalizacja populacji ────────────────────────────────────────────────

    def _init_population(self) -> list[Chromosome]:
        return [
            Chromosome(self.cfg.n_variables, self.cfg.bounds, self.cfg.precision)
            for _ in range(self.cfg.population_size)
        ]

    # ── Ocena populacji ────────────────────────────────────────────────────────

    def _evaluate(self, population: list[Chromosome]) -> None:
        for ind in population:
            if ind.fitness is None:
                ind.fitness = self.func(ind.decode())

    # ── Selekcja ──────────────────────────────────────────────────────────────

    def _select(self, population: list[Chromosome]) -> list[Chromosome]:
        n_select = self.cfg.population_size - self.cfg.elite_size
        sel_fn = SELECTION_METHODS[self.cfg.selection_method]
        kwargs = {}
        if self.cfg.selection_method == "tournament":
            kwargs["tournament_size"] = self.cfg.tournament_size
        return sel_fn(population, n_select, self.cfg.maximize, **kwargs)

    # ── Krzyżowanie ───────────────────────────────────────────────────────────

    def _crossover(self, parents: list[Chromosome]) -> list[Chromosome]:
        cross_fn = CROSSOVER_METHODS[self.cfg.crossover_method]
        offspring = []
        np.random.shuffle(parents)
        for i in range(0, len(parents) - 1, 2):
            if np.random.rand() < self.cfg.crossover_prob:
                c1, c2 = cross_fn(parents[i], parents[i + 1])
            else:
                c1, c2 = parents[i].copy(), parents[i + 1].copy()
            offspring.extend([c1, c2])
        if len(parents) % 2 == 1:
            offspring.append(parents[-1].copy())
        return offspring

    # ── Mutacja ───────────────────────────────────────────────────────────────

    def _mutate(self, population: list[Chromosome]) -> list[Chromosome]:
        mut_fn = MUTATION_METHODS[self.cfg.mutation_method]
        return [mut_fn(ind, self.cfg.mutation_prob) for ind in population]

    # ── Inwersja ──────────────────────────────────────────────────────────────

    def _invert(self, population: list[Chromosome]) -> list[Chromosome]:
        return [inversion(ind, self.cfg.inversion_prob) for ind in population]

    # ── Strategia elitarna ────────────────────────────────────────────────────

    def _get_elite(self, population: list[Chromosome]) -> list[Chromosome]:
        if self.cfg.elite_size <= 0:
            return []
        reverse = self.cfg.maximize
        sorted_pop = sorted(population, key=lambda c: c.fitness, reverse=reverse)
        return [sorted_pop[i].copy() for i in range(self.cfg.elite_size)]

    def _best(self, population: list[Chromosome]) -> Chromosome:
        reverse = self.cfg.maximize
        return sorted(population, key=lambda c: c.fitness, reverse=reverse)[0]

    # ── Główna pętla ──────────────────────────────────────────────────────────

    def run(self) -> GAResult:
        self._stop_flag = False
        start = time.perf_counter()

        population = self._init_population()
        self._evaluate(population)

        history_best: list[float] = []
        history_avg: list[float] = []

        for epoch in range(self.cfg.n_epochs):
            if self._stop_flag:
                break

            # Elita
            elite = self._get_elite(population)

            # Selekcja → krzyżowanie → mutacja → inwersja
            parents = self._select(population)
            offspring = self._crossover(parents)
            offspring = self._mutate(offspring)
            offspring = self._invert(offspring)

            # Nowa populacja = elita + potomkowie
            population = elite + offspring[: self.cfg.population_size - len(elite)]

            # Ocena nowej populacji
            self._evaluate(population)

            # Statystyki epoki
            fits = [c.fitness for c in population]
            best_fit = min(fits) if not self.cfg.maximize else max(fits)
            avg_fit = float(np.mean(fits))
            history_best.append(best_fit)
            history_avg.append(avg_fit)
            

            if self.progress_callback:
                self.progress_callback(epoch + 1, best_fit, avg_fit)

        elapsed = time.perf_counter() - start
        best_ind = self._best(population)

        return GAResult(
            best_individual=best_ind,
            best_fitness=best_ind.fitness,
            best_x=best_ind.decode(),
            history_best=history_best,
            history_avg=history_avg,
            elapsed_time=elapsed,
            n_epochs_run=epoch + 1,
        )

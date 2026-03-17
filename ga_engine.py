from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable


A = -2.048
B = 2.048


class Chromosome:
    """Jeden chromosom = jedna zmienna zakodowana binarnie."""

    def __init__(self, bits_count: int, bits: list[int] | None = None):
        self.bits_count = bits_count
        self.bits = bits[:] if bits is not None else [random.randint(0, 1) for _ in range(bits_count)]

    def copy(self) -> "Chromosome":
        return Chromosome(self.bits_count, self.bits)

    def decode(self, a: float = A, b: float = B) -> float:
        decimal = int("".join(str(bit) for bit in self.bits), 2)
        max_decimal = (2 ** self.bits_count) - 1
        return a + (decimal / max_decimal) * (b - a)


class Individual:
    """Osobnik składa się z N chromosomów."""

    def __init__(self, variables_count: int, bits_per_variable: int, chromosomes: list[Chromosome] | None = None):
        self.variables_count = variables_count
        self.bits_per_variable = bits_per_variable
        self.chromosomes = chromosomes[:] if chromosomes is not None else [
            Chromosome(bits_per_variable) for _ in range(variables_count)
        ]
        self.value: float | None = None
        self.fitness: float | None = None

    def copy(self) -> "Individual":
        copied = Individual(
            self.variables_count,
            self.bits_per_variable,
            [chrom.copy() for chrom in self.chromosomes],
        )
        copied.value = self.value
        copied.fitness = self.fitness
        return copied

    def decode(self) -> list[float]:
        return [chrom.decode() for chrom in self.chromosomes]

    def all_bits(self) -> list[int]:
        bits: list[int] = []
        for chrom in self.chromosomes:
            bits.extend(chrom.bits)
        return bits

    @classmethod
    def from_all_bits(cls, bits: list[int], variables_count: int, bits_per_variable: int) -> "Individual":
        chromosomes = []
        for i in range(variables_count):
            start = i * bits_per_variable
            end = start + bits_per_variable
            chromosomes.append(Chromosome(bits_per_variable, bits[start:end]))
        return cls(variables_count, bits_per_variable, chromosomes)


def rosenbrock(values: list[float]) -> float:
    total = 0.0
    for i in range(len(values) - 1):
        total += 100 * (values[i + 1] - values[i] ** 2) ** 2 + (values[i] - 1) ** 2
    return total


def value_to_fitness(value: float, maximize: bool) -> float:
    if maximize:
        return value
    return 1.0 / (1.0 + value)


def evaluate_population(population: list[Individual], objective_function: Callable[[list[float]], float], maximize: bool) -> None:
    for individual in population:
        values = individual.decode()
        individual.value = objective_function(values)
        individual.fitness = value_to_fitness(individual.value, maximize)


# -------------------- Selekcja --------------------

def select_best(population: list[Individual], count: int, maximize: bool) -> list[Individual]:
    key = (lambda ind: ind.value)
    sorted_population = sorted(population, key=key, reverse=maximize)
    return [ind.copy() for ind in sorted_population[:count]]


def select_tournament(population: list[Individual], count: int, maximize: bool, tournament_size: int = 3) -> list[Individual]:
    selected = []
    for _ in range(count):
        group = random.sample(population, tournament_size)
        winner = max(group, key=lambda ind: ind.value) if maximize else min(group, key=lambda ind: ind.value)
        selected.append(winner.copy())
    return selected


def select_roulette(population: list[Individual], count: int, maximize: bool) -> list[Individual]:
    if maximize:
        weights = [max(ind.value, 0.0) + 1e-9 for ind in population]
    else:
        weights = [ind.fitness for ind in population]
    chosen = random.choices(population, weights=weights, k=count)
    return [ind.copy() for ind in chosen]


# -------------------- Krzyżowanie --------------------

def crossover_single_point(parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
    bits1 = parent1.all_bits()
    bits2 = parent2.all_bits()
    point = random.randint(1, len(bits1) - 1)
    child1_bits = bits1[:point] + bits2[point:]
    child2_bits = bits2[:point] + bits1[point:]
    return (
        Individual.from_all_bits(child1_bits, parent1.variables_count, parent1.bits_per_variable),
        Individual.from_all_bits(child2_bits, parent1.variables_count, parent1.bits_per_variable),
    )


def crossover_two_point(parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
    bits1 = parent1.all_bits()
    bits2 = parent2.all_bits()
    a, b = sorted(random.sample(range(1, len(bits1)), 2))
    child1_bits = bits1[:a] + bits2[a:b] + bits1[b:]
    child2_bits = bits2[:a] + bits1[a:b] + bits2[b:]
    return (
        Individual.from_all_bits(child1_bits, parent1.variables_count, parent1.bits_per_variable),
        Individual.from_all_bits(child2_bits, parent1.variables_count, parent1.bits_per_variable),
    )


def crossover_uniform(parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
    bits1 = parent1.all_bits()
    bits2 = parent2.all_bits()
    child1_bits = []
    child2_bits = []
    for b1, b2 in zip(bits1, bits2):
        if random.random() < 0.5:
            child1_bits.append(b1)
            child2_bits.append(b2)
        else:
            child1_bits.append(b2)
            child2_bits.append(b1)
    return (
        Individual.from_all_bits(child1_bits, parent1.variables_count, parent1.bits_per_variable),
        Individual.from_all_bits(child2_bits, parent1.variables_count, parent1.bits_per_variable),
    )


def crossover_granular(parent1: Individual, parent2: Individual, grain_size: int = 4) -> tuple[Individual, Individual]:
    bits1 = parent1.all_bits()
    bits2 = parent2.all_bits()
    child1_bits = bits1[:]
    child2_bits = bits2[:]
    for start in range(0, len(bits1), grain_size):
        end = min(start + grain_size, len(bits1))
        if random.random() < 0.5:
            child1_bits[start:end] = bits2[start:end]
            child2_bits[start:end] = bits1[start:end]
    return (
        Individual.from_all_bits(child1_bits, parent1.variables_count, parent1.bits_per_variable),
        Individual.from_all_bits(child2_bits, parent1.variables_count, parent1.bits_per_variable),
    )


# -------------------- Mutacje i inwersja --------------------

def mutation_edge(individual: Individual, probability: float) -> Individual:
    child = individual.copy()
    if random.random() < probability:
        chrom_index = random.randrange(child.variables_count)
        if random.random() < 0.5:
            bit_index = 0
        else:
            bit_index = child.bits_per_variable - 1
        child.chromosomes[chrom_index].bits[bit_index] ^= 1
    return child



def mutation_single_point(individual: Individual, probability: float) -> Individual:
    child = individual.copy()
    if random.random() < probability:
        chrom_index = random.randrange(child.variables_count)
        bit_index = random.randrange(child.bits_per_variable)
        child.chromosomes[chrom_index].bits[bit_index] ^= 1
    return child



def mutation_two_point(individual: Individual, probability: float) -> Individual:
    child = individual.copy()
    if random.random() < probability:
        all_bits = child.all_bits()
        first, second = random.sample(range(len(all_bits)), 2)
        all_bits[first] ^= 1
        all_bits[second] ^= 1
        child = Individual.from_all_bits(all_bits, child.variables_count, child.bits_per_variable)
    return child



def inversion(individual: Individual, probability: float) -> Individual:
    child = individual.copy()
    if random.random() < probability:
        all_bits = child.all_bits()
        a, b = sorted(random.sample(range(len(all_bits)), 2))
        all_bits[a:b] = list(reversed(all_bits[a:b]))
        child = Individual.from_all_bits(all_bits, child.variables_count, child.bits_per_variable)
    return child


@dataclass
class GAConfig:
    population_size: int = 50
    epochs: int = 100
    variables_count: int = 2
    bits_per_variable: int = 16
    selection_method: str = "tournament"
    crossover_method: str = "single_point"
    mutation_method: str = "single_point"
    crossover_probability: float = 0.8
    mutation_probability: float = 0.02
    inversion_probability: float = 0.01
    elite_size: int = 1
    tournament_size: int = 3
    maximize: bool = False


@dataclass
class GAResult:
    best_value: float
    best_fitness: float
    best_vector: list[float]
    history_best: list[float]
    history_avg: list[float]
    elapsed_time: float


SELECTIONS = {
    "best": select_best,
    "roulette": select_roulette,
    "tournament": select_tournament,
}

CROSSOVERS = {
    "single_point": crossover_single_point,
    "two_point": crossover_two_point,
    "uniform": crossover_uniform,
    "granular": crossover_granular,
}

MUTATIONS = {
    "edge": mutation_edge,
    "single_point": mutation_single_point,
    "two_point": mutation_two_point,
}


class GeneticAlgorithm:
    def __init__(self, config: GAConfig, progress_callback: Callable[[int, float, float], None] | None = None):
        self.config = config
        self.progress_callback = progress_callback
        self.should_stop = False

    def stop(self) -> None:
        self.should_stop = True

    def _create_population(self) -> list[Individual]:
        return [Individual(self.config.variables_count, self.config.bits_per_variable) for _ in range(self.config.population_size)]

    def _get_elite(self, population: list[Individual]) -> list[Individual]:
        if self.config.elite_size <= 0:
            return []
        ordered = sorted(population, key=lambda ind: ind.value, reverse=self.config.maximize)
        return [ind.copy() for ind in ordered[: self.config.elite_size]]

    def _select(self, population: list[Individual]) -> list[Individual]:
        count = self.config.population_size - self.config.elite_size
        selection = SELECTIONS[self.config.selection_method]
        if self.config.selection_method == "tournament":
            return selection(population, count, self.config.maximize, self.config.tournament_size)
        return selection(population, count, self.config.maximize)

    def _crossover(self, parents: list[Individual]) -> list[Individual]:
        random.shuffle(parents)
        crossover = CROSSOVERS[self.config.crossover_method]
        children: list[Individual] = []
        for i in range(0, len(parents) - 1, 2):
            p1 = parents[i]
            p2 = parents[i + 1]
            if random.random() < self.config.crossover_probability:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            children.extend([c1, c2])
        if len(parents) % 2 == 1:
            children.append(parents[-1].copy())
        return children

    def _mutate(self, children: list[Individual]) -> list[Individual]:
        mutation = MUTATIONS[self.config.mutation_method]
        return [mutation(child, self.config.mutation_probability) for child in children]

    def _invert(self, children: list[Individual]) -> list[Individual]:
        return [inversion(child, self.config.inversion_probability) for child in children]

    def run(self) -> GAResult:
        start = time.perf_counter()
        population = self._create_population()
        evaluate_population(population, rosenbrock, self.config.maximize)

        history_best: list[float] = []
        history_avg: list[float] = []

        for epoch in range(1, self.config.epochs + 1):
            if self.should_stop:
                break

            elite = self._get_elite(population)
            parents = self._select(population)
            children = self._crossover(parents)
            children = self._mutate(children)
            children = self._invert(children)

            population = elite + children[: self.config.population_size - len(elite)]
            evaluate_population(population, rosenbrock, self.config.maximize)

            values = [ind.value for ind in population]
            best_value = max(values) if self.config.maximize else min(values)
            avg_value = sum(values) / len(values)
            history_best.append(best_value)
            history_avg.append(avg_value)

            if self.progress_callback is not None:
                self.progress_callback(epoch, best_value, avg_value)

        best_individual = max(population, key=lambda ind: ind.value) if self.config.maximize else min(population, key=lambda ind: ind.value)
        elapsed = time.perf_counter() - start

        return GAResult(
            best_value=best_individual.value,
            best_fitness=best_individual.fitness,
            best_vector=best_individual.decode(),
            history_best=history_best,
            history_avg=history_avg,
            elapsed_time=elapsed,
        )

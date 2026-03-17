"""
Binarna reprezentacja chromosomu z konfiguracją dokładności.
"""
import numpy as np


class Chromosome:
    """
    Chromosom z binarną reprezentacją zmiennych rzeczywistych.
    Każda zmienna kodowana jest jako ciąg bitów z zadaną dokładnością.
    """

    def __init__(self, n_variables: int, bounds: list[tuple], precision: int = 6):
        """
        n_variables: liczba zmiennych funkcji
        bounds: lista krotek (min, max) dla każdej zmiennej
        precision: liczba miejsc po przecinku (dokładność)
        """
        self.n_variables = n_variables
        self.bounds = bounds if len(bounds) == n_variables else [bounds[0]] * n_variables
        self.precision = precision

        # Liczba bitów potrzebna do zakodowania każdej zmiennej
        self.bits_per_var = self._calc_bits()
        self.total_bits = self.bits_per_var * n_variables

        # Losowa inicjalizacja bitów
        self.genes = np.random.randint(0, 2, self.total_bits, dtype=np.uint8)
        self.fitness: float = None

    def _calc_bits(self) -> int:
        """Oblicza minimalną liczbę bitów dla zadanej dokładności."""
        max_range = max(hi - lo for lo, hi in self.bounds)
        n_values = max_range * (10 ** self.precision)
        return int(np.ceil(np.log2(n_values + 1)))

    def decode(self) -> np.ndarray:
        """Dekoduje chromosom do wartości rzeczywistych."""
        values = np.empty(self.n_variables)
        for i in range(self.n_variables):
            lo, hi = self.bounds[i]
            bits = self.genes[i * self.bits_per_var:(i + 1) * self.bits_per_var]
            # Konwersja binarnie → int → float w zakresie [lo, hi]
            int_val = bits.dot(1 << np.arange(self.bits_per_var - 1, -1, -1))
            max_int = (1 << self.bits_per_var) - 1
            values[i] = lo + int_val / max_int * (hi - lo)
        return values

    @classmethod
    def from_genes(cls, genes: np.ndarray, n_variables: int,
                   bounds: list[tuple], precision: int = 6) -> "Chromosome":
        """Tworzy chromosom z gotowego ciągu bitów."""
        c = cls.__new__(cls)
        c.n_variables = n_variables
        c.bounds = bounds if len(bounds) == n_variables else [bounds[0]] * n_variables
        c.precision = precision
        c.bits_per_var = c._calc_bits()
        c.total_bits = c.bits_per_var * n_variables
        c.genes = genes.copy()
        c.fitness = None
        return c

    def copy(self) -> "Chromosome":
        return Chromosome.from_genes(self.genes, self.n_variables,
                                     self.bounds, self.precision)

    def __repr__(self):
        vals = self.decode()
        return f"Chromosome(fitness={self.fitness:.6f}, x={vals})"

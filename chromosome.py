import numpy as np

class Chromosome:
    def __init__(self, number_of_bits, random_init=True):
        self.number_of_bits = number_of_bits [cite: 19]
        # Inicjalizacja losowym ciągiem bitów (0 i 1)
        if random_init:
            self.bits = np.random.randint(2, size=self.number_of_bits) [cite: 19]
        else:
            self.bits = np.zeros(self.number_of_bits)

    def decode(self, a, b):
        """Zamiana binarnego chromosomu na liczbę rzeczywistą z zakresu [a, b]"""
        # Obliczanie wartości dziesiętnej z bitów
        decimal_val = int("".join(map(str, self.bits)), 2)
        # Mapowanie na zakres [a, b]
        return a + decimal_val * (b - a) / (2**self.number_of_bits - 1)
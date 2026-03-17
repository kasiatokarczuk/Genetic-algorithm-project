"""
Testy bez GUI – sprawdzenie logiki algorytmu.
Uruchom:  python tests/test_ga.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import GeneticAlgorithm, GAConfig, RosenbrockFunction, save_results


def test_rosenbrock_2d():
    print("=" * 50)
    print("Test: Rosenbrock 2D – minimalizacja")
    func = RosenbrockFunction(n_variables=2)
    config = GAConfig(
        population_size=100,
        n_epochs=200,
        n_variables=2,
        bounds=func.bounds,
        precision=6,
        selection_method="tournament",
        crossover_method="single_point",
        mutation_method="single_point",
        mutation_prob=0.01,
        elite_size=2,
        maximize=False,
    )
    ga = GeneticAlgorithm(func, config)
    result = ga.run()
    print(f"Najlepszy wynik:  {result.best_fitness:.8f}  (oczekiwane: ~0.0)")
    print(f"Najlepsze x:      {result.best_x}")
    print(f"Oczekiwane x:     {func.global_minimum_x}")
    print(f"Czas:             {result.elapsed_time:.3f} s")
    assert result.best_fitness < 1.0, "Algorytm nie zbiegł wystarczająco blisko minimum!"
    print("✓ Test 2D zaliczony.\n")


def test_rosenbrock_5d():
    print("=" * 50)
    print("Test: Rosenbrock 5D – minimalizacja")
    func = RosenbrockFunction(n_variables=5)
    config = GAConfig(
        population_size=200,
        n_epochs=500,
        n_variables=5,
        bounds=func.bounds,
        precision=4,
        selection_method="roulette",
        crossover_method="two_point",
        mutation_method="edge",
        mutation_prob=0.02,
        elite_size=3,
        maximize=False,
    )
    ga = GeneticAlgorithm(func, config)
    result = ga.run()
    print(f"Najlepszy wynik:  {result.best_fitness:.6f}")
    print(f"Czas:             {result.elapsed_time:.3f} s")
    print("✓ Test 5D zakończony.\n")


def test_save_results():
    print("=" * 50)
    print("Test: Zapis wyników")
    func = RosenbrockFunction(n_variables=2)
    config = GAConfig(n_variables=2, bounds=func.bounds, n_epochs=50, population_size=30)
    ga = GeneticAlgorithm(func, config)
    result = ga.run()
    paths = save_results(result, config, "rosenbrock", output_dir="results/test")
    print(f"CSV:  {paths['csv']}")
    print(f"JSON: {paths['json']}")
    assert os.path.exists(paths["csv"])
    assert os.path.exists(paths["json"])
    print("✓ Zapis wyników OK.\n")


if __name__ == "__main__":
    test_rosenbrock_2d()
    test_rosenbrock_5d()
    test_save_results()
    print("Wszystkie testy zaliczone ✓")

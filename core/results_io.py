"""
Zapisywanie wyników do pliku CSV i JSON.
"""
import csv
import json
import os
from datetime import datetime
from .genetic_algorithm import GAResult, GAConfig


def save_results(result: GAResult, config: GAConfig,
                 func_name: str, output_dir: str = "results") -> dict[str, str]:
    """
    Zapisuje wyniki do pliku CSV (historia) i JSON (podsumowanie).
    Zwraca słownik z ścieżkami do plików.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{func_name}_{config.n_variables}var_{timestamp}"

    # ── Historia epok → CSV ───────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, f"{base}_history.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "best_fitness", "avg_fitness"])
        for epoch, (best, avg) in enumerate(
                zip(result.history_best, result.history_avg), start=1):
            writer.writerow([epoch, best, avg])

    # ── Podsumowanie → JSON ───────────────────────────────────────────────────
    summary = {
        "timestamp": timestamp,
        "function": func_name,
        "config": {
            "n_variables": config.n_variables,
            "population_size": config.population_size,
            "n_epochs": config.n_epochs,
            "precision": config.precision,
            "selection": config.selection_method,
            "crossover": config.crossover_method,
            "mutation": config.mutation_method,
            "crossover_prob": config.crossover_prob,
            "mutation_prob": config.mutation_prob,
            "inversion_prob": config.inversion_prob,
            "elite_size": config.elite_size,
            "maximize": config.maximize,
        },
        "result": {
            "best_fitness": result.best_fitness,
            "best_x": result.best_x.tolist(),
            "elapsed_time_s": round(result.elapsed_time, 4),
            "epochs_run": result.n_epochs_run,
        },
    }
    json_path = os.path.join(output_dir, f"{base}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {"csv": csv_path, "json": json_path}

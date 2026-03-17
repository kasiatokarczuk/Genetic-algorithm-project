from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from ga_engine import GAConfig, GAResult


def save_results(result: GAResult, config: GAConfig, output_dir: str = "results") -> tuple[Path, Path]:
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = folder / f"ga_history_{timestamp}.csv"
    json_path = folder / f"ga_summary_{timestamp}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "best_value", "average_value"])
        for index, (best, avg) in enumerate(zip(result.history_best, result.history_avg), start=1):
            writer.writerow([index, best, avg])

    summary = {
        "config": config.__dict__,
        "best_value": result.best_value,
        "best_fitness": result.best_fitness,
        "best_vector": result.best_vector,
        "elapsed_time": result.elapsed_time,
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    return csv_path, json_path

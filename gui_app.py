from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ga_engine import GAConfig, GeneticAlgorithm, SELECTIONS, CROSSOVERS, MUTATIONS
from results_io import save_results


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Uproszczony algorytm genetyczny - Rosenbrock")
        self.geometry("1000x650")

        self.ga: GeneticAlgorithm | None = None
        self.result = None
        self.history_best: list[float] = []
        self.history_avg: list[float] = []

        self._build_ui()

    def _build_ui(self):
        left = tk.Frame(self, padx=10, pady=10)
        left.pack(side="left", fill="y")

        right = tk.Frame(self, padx=10, pady=10)
        right.pack(side="right", fill="both", expand=True)

        self.population_var = tk.IntVar(value=50)
        self.epochs_var = tk.IntVar(value=100)
        self.variables_var = tk.IntVar(value=2)
        self.bits_var = tk.IntVar(value=16)
        self.selection_var = tk.StringVar(value="tournament")
        self.crossover_var = tk.StringVar(value="single_point")
        self.mutation_var = tk.StringVar(value="single_point")
        self.crossover_prob_var = tk.DoubleVar(value=0.8)
        self.mutation_prob_var = tk.DoubleVar(value=0.02)
        self.inversion_prob_var = tk.DoubleVar(value=0.01)
        self.elite_var = tk.IntVar(value=1)
        self.tournament_var = tk.IntVar(value=3)
        self.goal_var = tk.StringVar(value="minimize")

        def add_spin(label: str, variable, from_, to, increment=1):
            tk.Label(left, text=label).pack(anchor="w")
            ttk.Spinbox(left, from_=from_, to=to, increment=increment, textvariable=variable, width=18).pack(anchor="w", pady=(0, 8))

        tk.Label(left, text="Parametry", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 8))
        add_spin("Wielkość populacji", self.population_var, 10, 500)
        add_spin("Liczba epok", self.epochs_var, 1, 5000)
        add_spin("Liczba zmiennych N", self.variables_var, 2, 50)
        add_spin("Bity na zmienną", self.bits_var, 4, 32)

        tk.Label(left, text="Selekcja").pack(anchor="w")
        ttk.Combobox(left, textvariable=self.selection_var, values=list(SELECTIONS.keys()), state="readonly", width=16).pack(anchor="w", pady=(0, 8))

        tk.Label(left, text="Krzyżowanie").pack(anchor="w")
        ttk.Combobox(left, textvariable=self.crossover_var, values=list(CROSSOVERS.keys()), state="readonly", width=16).pack(anchor="w", pady=(0, 8))

        tk.Label(left, text="Mutacja").pack(anchor="w")
        ttk.Combobox(left, textvariable=self.mutation_var, values=list(MUTATIONS.keys()), state="readonly", width=16).pack(anchor="w", pady=(0, 8))

        add_spin("Prawd. krzyżowania", self.crossover_prob_var, 0.0, 1.0, 0.01)
        add_spin("Prawd. mutacji", self.mutation_prob_var, 0.0, 1.0, 0.01)
        add_spin("Prawd. inwersji", self.inversion_prob_var, 0.0, 1.0, 0.01)
        add_spin("Elita", self.elite_var, 0, 20)
        add_spin("Rozmiar turnieju", self.tournament_var, 2, 20)

        tk.Label(left, text="Cel").pack(anchor="w")
        ttk.Combobox(left, textvariable=self.goal_var, values=["minimize", "maximize"], state="readonly", width=16).pack(anchor="w", pady=(0, 12))

        tk.Button(left, text="Start", command=self.start_algorithm).pack(fill="x", pady=2)
        tk.Button(left, text="Stop", command=self.stop_algorithm).pack(fill="x", pady=2)
        tk.Button(left, text="Zapisz wyniki", command=self.save_current_result).pack(fill="x", pady=2)

        self.status_label = tk.Label(left, text="Gotowe")
        self.status_label.pack(anchor="w", pady=(12, 4))
        self.time_label = tk.Label(left, text="Czas: -")
        self.time_label.pack(anchor="w")
        self.best_label = tk.Label(left, text="Najlepszy wynik: -")
        self.best_label.pack(anchor="w")

        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self.axis.set_title("Wartość funkcji w kolejnych epokach")
        self.axis.set_xlabel("Epoka")
        self.axis.set_ylabel("Wartość")
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

    def _build_config(self) -> GAConfig:
        return GAConfig(
            population_size=self.population_var.get(),
            epochs=self.epochs_var.get(),
            variables_count=self.variables_var.get(),
            bits_per_variable=self.bits_var.get(),
            selection_method=self.selection_var.get(),
            crossover_method=self.crossover_var.get(),
            mutation_method=self.mutation_var.get(),
            crossover_probability=self.crossover_prob_var.get(),
            mutation_probability=self.mutation_prob_var.get(),
            inversion_probability=self.inversion_prob_var.get(),
            elite_size=self.elite_var.get(),
            tournament_size=self.tournament_var.get(),
            maximize=self.goal_var.get() == "maximize",
        )

    def start_algorithm(self):
        self.history_best = []
        self.history_avg = []
        self._clear_plot()
        config = self._build_config()
        self.ga = GeneticAlgorithm(config, progress_callback=self._on_progress)
        self.status_label.config(text="Trwa obliczanie...")
        thread = threading.Thread(target=self._run_worker, daemon=True)
        thread.start()

    def _run_worker(self):
        if self.ga is None:
            return
        self.result = self.ga.run()
        self.after(0, self._on_done)

    def _on_progress(self, epoch: int, best_value: float, average_value: float):
        self.history_best.append(best_value)
        self.history_avg.append(average_value)
        self.after(0, self._refresh_plot)
        self.after(0, lambda: self.best_label.config(text=f"Najlepszy wynik: {best_value:.6f}"))
        self.after(0, lambda: self.status_label.config(text=f"Epoka {epoch}"))

    def _clear_plot(self):
        self.axis.clear()
        self.axis.set_title("Wartość funkcji w kolejnych epokach")
        self.axis.set_xlabel("Epoka")
        self.axis.set_ylabel("Wartość")
        self.canvas.draw()

    def _refresh_plot(self):
        self.axis.clear()
        self.axis.set_title("Wartość funkcji w kolejnych epokach")
        self.axis.set_xlabel("Epoka")
        self.axis.set_ylabel("Wartość")
        if self.history_best:
            self.axis.plot(range(1, len(self.history_best) + 1), self.history_best, label="best")
        if self.history_avg:
            self.axis.plot(range(1, len(self.history_avg) + 1), self.history_avg, label="avg")
        self.axis.legend()
        self.canvas.draw()

    def _on_done(self):
        if self.result is None:
            return
        self.status_label.config(text="Zakończono")
        self.time_label.config(text=f"Czas: {self.result.elapsed_time:.3f} s")
        self.best_label.config(text=f"Najlepszy wynik: {self.result.best_value:.6f}")
        self._refresh_plot()

    def stop_algorithm(self):
        if self.ga is not None:
            self.ga.stop()
            self.status_label.config(text="Zatrzymywanie...")

    def save_current_result(self):
        if self.result is None or self.ga is None:
            messagebox.showwarning("Brak wyniku", "Najpierw uruchom algorytm.")
            return
        folder = filedialog.askdirectory(title="Wybierz folder")
        if not folder:
            return
        csv_path, json_path = save_results(self.result, self.ga.config, folder)
        messagebox.showinfo("Zapisano", f"Zapisano:\n{csv_path}\n{json_path}")

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
        self.title("Algorytm genetyczny - Rosenbrock")
        self.geometry("1000x700")
        self.configure(bg="#f3ede3")

        self.colors = {
            "bg_main": "#f3ede3",
            "bg_sidebar": "#ebe1d2",
            "bg_card": "#fffaf3",
            "bg_input": "#f8f1e7",
            "border": "#d3c3af",
            "text": "#40352c",
            "muted": "#7a6a5a",
            "accent": "#b38b6d",
            "accent_dark": "#9a7457",
            "accent_soft": "#dcc8b3",
            "plot_best": "#8377d5" ,
            "plot_avg": "#b29d8a",
        }

        self.ga: GeneticAlgorithm | None = None
        self.result = None
        self.history_best: list[float] = []
        self.history_avg: list[float] = []

        self._setup_styles()
        self._build_ui()

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel", background=self.colors["bg_sidebar"], foreground=self.colors["text"], font=("Segoe UI", 10))
        style.configure("TCombobox", fieldbackground=self.colors["bg_input"], background=self.colors["bg_input"], foreground=self.colors["text"], bordercolor=self.colors["border"], lightcolor=self.colors["border"], darkcolor=self.colors["border"], arrowcolor=self.colors["accent_dark"], padding=5)
        style.map("TCombobox", fieldbackground=[("readonly", self.colors["bg_input"])], selectbackground=[("readonly", self.colors["bg_input"])], selectforeground=[("readonly", self.colors["text"])])
        style.configure("TSpinbox", fieldbackground=self.colors["bg_input"], background=self.colors["bg_input"], foreground=self.colors["text"], bordercolor=self.colors["border"], lightcolor=self.colors["border"], darkcolor=self.colors["border"], arrowsize=12, padding=5)
        style.configure("Vertical.TScrollbar", background=self.colors["accent_soft"], troughcolor=self.colors["bg_sidebar"], bordercolor=self.colors["bg_sidebar"], arrowcolor=self.colors["accent_dark"])

    def _build_ui(self):
        # --- Lewa kolumna z przewijaniem ---
        left_outer = tk.Frame(self, width=260, bg=self.colors["bg_sidebar"], highlightthickness=1, highlightbackground=self.colors["border"])
        left_outer.pack(side="left", fill="y")
        left_outer.pack_propagate(False)

        scroll_canvas = tk.Canvas(left_outer, highlightthickness=0, bg=self.colors["bg_sidebar"], bd=0)
        scrollbar = ttk.Scrollbar(left_outer, orient="vertical", command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        scroll_canvas.pack(side="left", fill="both", expand=True)

        left = tk.Frame(scroll_canvas, padx=12, pady=12, bg=self.colors["bg_sidebar"])
        left_window = scroll_canvas.create_window((0, 0), window=left, anchor="nw")

        def on_frame_configure(event):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        def on_canvas_configure(event):
            scroll_canvas.itemconfig(left_window, width=event.width)

        left.bind("<Configure>", on_frame_configure)
        scroll_canvas.bind("<Configure>", on_canvas_configure)

        def bind_mousewheel(event):
            scroll_canvas.bind_all("<MouseWheel>", on_mousewheel)

        def unbind_mousewheel(event):
            scroll_canvas.unbind_all("<MouseWheel>")

        def on_mousewheel(event):
            scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        scroll_canvas.bind("<Enter>", bind_mousewheel)
        scroll_canvas.bind("<Leave>", unbind_mousewheel)
        left.bind("<Enter>", bind_mousewheel)
        left.bind("<Leave>", unbind_mousewheel)

        # --- Prawa strona ---
        right = tk.Frame(self, padx=10, pady=10, bg=self.colors["bg_main"])
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
            tk.Label(left, text=label, bg=self.colors["bg_sidebar"], fg=self.colors["text"], font=("Segoe UI", 10)).pack(anchor="w")
            ttk.Spinbox(left, from_=from_, to=to, increment=increment, textvariable=variable).pack(fill="x", pady=(2, 5), ipady=1)

        tk.Label(left, text="Parametry", font=("Segoe UI", 12, "bold"), bg=self.colors["bg_sidebar"], fg=self.colors["accent_dark"]).pack(anchor="w", pady=(0, 10))
        add_spin("Wielkość populacji", self.population_var, 10, 500)
        add_spin("Liczba epok", self.epochs_var, 1, 5000)
        add_spin("Liczba zmiennych N", self.variables_var, 2, 50)
        add_spin("Bity na zmienną", self.bits_var, 4, 32)

        tk.Label(left, text="Selekcja", bg=self.colors["bg_sidebar"], fg=self.colors["text"], font=("Segoe UI", 9)).pack(anchor="w")
        ttk.Combobox(left, textvariable=self.selection_var, values=list(SELECTIONS.keys()), state="readonly").pack(fill="x", pady=(2, 5), ipady=1)

        tk.Label(left, text="Krzyżowanie", bg=self.colors["bg_sidebar"], fg=self.colors["text"], font=("Segoe UI", 9)).pack(anchor="w")
        ttk.Combobox(left, textvariable=self.crossover_var, values=list(CROSSOVERS.keys()), state="readonly").pack(fill="x", pady=(2, 5), ipady=1)

        tk.Label(left, text="Mutacja", bg=self.colors["bg_sidebar"], fg=self.colors["text"], font=("Segoe UI", 9)).pack(anchor="w")
        ttk.Combobox(left, textvariable=self.mutation_var, values=list(MUTATIONS.keys()), state="readonly").pack(fill="x", pady=(2, 5), ipady=1)

        add_spin("Prawd. krzyżowania", self.crossover_prob_var, 0.0, 1.0, 0.01)
        add_spin("Prawd. mutacji", self.mutation_prob_var, 0.0, 1.0, 0.01)
        add_spin("Prawd. inwersji", self.inversion_prob_var, 0.0, 1.0, 0.01)
        add_spin("Elita", self.elite_var, 0, 20)
        add_spin("Rozmiar turnieju", self.tournament_var, 2, 20)

        tk.Label(left, text="Cel", bg=self.colors["bg_sidebar"], fg=self.colors["text"], font=("Segoe UI", 9)).pack(anchor="w")
        ttk.Combobox(left, textvariable=self.goal_var, values=["minimize", "maximize"], state="readonly").pack(fill="x", pady=(3, 12), ipady=2)

        tk.Button(left, text="Start", command=self.start_algorithm, bg=self.colors["accent"], fg="white", activebackground=self.colors["accent_dark"], activeforeground="white", font=("Segoe UI", 10, "bold"), relief="flat", bd=0, cursor="hand2", pady=8).pack(fill="x", pady=2)
        tk.Button(left, text="Stop", command=self.stop_algorithm, bg=self.colors["accent"], fg="white", activebackground=self.colors["accent_dark"], activeforeground="white", font=("Segoe UI", 10, "bold"), relief="flat", bd=0, cursor="hand2", pady=8).pack(fill="x", pady=2)
        tk.Button(left, text="Zapisz wyniki", command=self.save_current_result, bg=self.colors["accent_soft"], fg=self.colors["text"], activebackground="#cfb79d", activeforeground=self.colors["text"], font=("Segoe UI", 10, "bold"), relief="flat", bd=0, cursor="hand2", pady=8).pack(fill="x", pady=2)

        # --- Pasek info nad wykresem ---
        info_bar = tk.Frame(right, bg=self.colors["bg_card"], highlightthickness=1, highlightbackground=self.colors["border"], padx=10, pady=8)
        info_bar.pack(fill="x", pady=(0, 6))

        tk.Label(info_bar, text="Status:", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["accent_dark"]).pack(side="left", padx=(0, 4))
        self.status_plot_label = tk.Label(info_bar, text="Gotowe", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text"])
        self.status_plot_label.pack(side="left", padx=(0, 20))

        tk.Label(info_bar, text="Czas:", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["accent_dark"]).pack(side="left", padx=(0, 4))
        self.time_plot_label = tk.Label(info_bar, text="—", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text"])
        self.time_plot_label.pack(side="left", padx=(0, 20))

        tk.Label(info_bar, text="Najlepszy wynik:", font=("Segoe UI", 9, "bold"), bg=self.colors["bg_card"], fg=self.colors["accent_dark"]).pack(side="left", padx=(0, 4))
        self.best_plot_label = tk.Label(info_bar, text="—", font=("Segoe UI", 9), bg=self.colors["bg_card"], fg=self.colors["text"])
        self.best_plot_label.pack(side="left")

        plot_frame = tk.Frame(right, bg=self.colors["bg_card"], highlightthickness=1, highlightbackground=self.colors["border"])
        plot_frame.pack(fill="both", expand=True)

        self.figure = Figure(figsize=(6, 4), dpi=100, facecolor=self.colors["bg_card"])
        self.axis = self.figure.add_subplot(111)
        self.axis.set_facecolor(self.colors["bg_card"])
        self.axis.set_title("Wartość funkcji w kolejnych epokach", color=self.colors["text"])
        self.axis.set_xlabel("Epoka", color=self.colors["muted"])
        self.axis.set_ylabel("Wartość", color=self.colors["muted"])
        self.axis.grid(True, alpha=0.25, color=self.colors["border"])
        for spine in self.axis.spines.values():
            spine.set_color(self.colors["border"])
        self.axis.tick_params(colors=self.colors["muted"])

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
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
        self.result = None
        self._clear_plot()

        self.status_plot_label.config(text="Trwa obliczanie...")
        self.time_plot_label.config(text="—")
        self.best_plot_label.config(text="—")

        config = self._build_config()
        self.ga = GeneticAlgorithm(config, progress_callback=self._on_progress)
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
        self.after(0, lambda: self.best_plot_label.config(text=f"{best_value:.6f}"))
        self.after(0, lambda: self.status_plot_label.config(text=f"Epoka {epoch}"))

    def _clear_plot(self):
        self.axis.clear()
        self.axis.set_facecolor(self.colors["bg_card"])
        self.axis.set_title("Wartość funkcji w kolejnych epokach", color=self.colors["text"])
        self.axis.set_xlabel("Epoka", color=self.colors["muted"])
        self.axis.set_ylabel("Wartość", color=self.colors["muted"])
        self.axis.grid(True, alpha=0.25, color=self.colors["border"])
        for spine in self.axis.spines.values():
            spine.set_color(self.colors["border"])
        self.axis.tick_params(colors=self.colors["muted"])
        self.canvas.draw()

    def _refresh_plot(self):
        self.axis.clear()
        self.axis.set_facecolor(self.colors["bg_card"])
        self.axis.set_title("Wartość funkcji w kolejnych epokach", color=self.colors["text"])
        self.axis.set_xlabel("Epoka", color=self.colors["muted"])
        self.axis.set_ylabel("Wartość", color=self.colors["muted"])
        self.axis.grid(True, alpha=0.25, color=self.colors["border"])
        for spine in self.axis.spines.values():
            spine.set_color(self.colors["border"])
        self.axis.tick_params(colors=self.colors["muted"])
        if self.history_best:
            self.axis.plot(range(1, len(self.history_best) + 1), self.history_best, label="best", color=self.colors["plot_best"], linewidth=2.2)
        if self.history_avg:
            self.axis.plot(range(1, len(self.history_avg) + 1), self.history_avg, label="avg", color=self.colors["plot_avg"], linewidth=2.2)
        legend = self.axis.legend(frameon=True)
        if legend:
            legend.get_frame().set_facecolor(self.colors["bg_card"])
            legend.get_frame().set_edgecolor(self.colors["border"])
        self.canvas.draw()

    def _on_done(self):
        if self.result is None:
            return
        self.status_plot_label.config(text="Zakończono")
        self.time_plot_label.config(text=f"{self.result.elapsed_time:.3f} s")
        self.best_plot_label.config(text=f"{self.result.best_value:.6f}")
        self._refresh_plot()

    def stop_algorithm(self):
        if self.ga is not None:
            self.ga.stop()
            self.status_plot_label.config(text="Zatrzymywanie...")

    def save_current_result(self):
        if self.result is None or self.ga is None:
            messagebox.showwarning("Brak wyniku", "Najpierw uruchom algorytm.")
            return
        folder = filedialog.askdirectory(title="Wybierz folder")
        if not folder:
            return
        csv_path, json_path = save_results(self.result, self.ga.config, folder)
        messagebox.showinfo("Zapisano", f"Zapisano:\n{csv_path}\n{json_path}")
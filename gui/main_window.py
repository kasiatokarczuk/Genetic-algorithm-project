"""
Główne okno aplikacji GUI (tkinter).
Osoba odpowiedzialna za GUI uruchamia tę część.
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import sys
import os

# Umożliwia uruchomienie GUI bezpośrednio z folderu gui/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import (
    GeneticAlgorithm, GAConfig, RosenbrockFunction,
    TEST_FUNCTIONS, SELECTION_METHODS, CROSSOVER_METHODS,
    MUTATION_METHODS, save_results,
)
from gui.plot_panel import PlotPanel


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Algorytm Genetyczny – Optymalizacja Funkcji")
        self.resizable(True, True)
        self.geometry("1100x700")
        self.configure(bg="#1e1e2e")

        self._ga: GeneticAlgorithm | None = None
        self._result = None
        self._thread: threading.Thread | None = None

        self._build_ui()

    # ── Budowanie interfejsu ───────────────────────────────────────────────────

    def _build_ui(self):
        # Lewy panel: konfiguracja
        left = tk.Frame(self, bg="#1e1e2e", width=320)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        left.pack_propagate(False)

        self._build_config_panel(left)

        # Prawy panel: wykresy
        right = tk.Frame(self, bg="#1e1e2e")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=10)

        self.plot_panel = PlotPanel(right)
        self.plot_panel.pack(fill=tk.BOTH, expand=True)

    def _label(self, parent, text, **kw):
        return tk.Label(parent, text=text, bg="#1e1e2e", fg="#cdd6f4",
                        font=("Segoe UI", 9), **kw)

    def _section(self, parent, text):
        tk.Label(parent, text=text, bg="#313244", fg="#89b4fa",
                 font=("Segoe UI", 9, "bold"),
                 anchor="w", padx=6).pack(fill=tk.X, pady=(8, 2))

    def _build_config_panel(self, parent):
        # Tytuł
        tk.Label(parent, text="⚙  Konfiguracja GA", bg="#1e1e2e",
                 fg="#cba6f7", font=("Segoe UI", 12, "bold")).pack(pady=(0, 6))

        scroll_frame = _ScrollFrame(parent)
        scroll_frame.pack(fill=tk.BOTH, expand=True)
        f = scroll_frame.inner

        # ── Funkcja testowa ───────────────────────────────────────────────────
        self._section(f, "Funkcja testowa")
        self.var_func = tk.StringVar(value="rosenbrock")
        ttk.Combobox(f, textvariable=self.var_func,
                     values=list(TEST_FUNCTIONS.keys()), state="readonly",
                     width=26).pack(padx=6, pady=2, fill=tk.X)

        self._label(f, "Liczba zmiennych (N):").pack(anchor="w", padx=6)
        self.var_n = tk.IntVar(value=2)
        ttk.Spinbox(f, from_=2, to=50, textvariable=self.var_n,
                    width=10).pack(padx=6, pady=2, fill=tk.X)

        self._label(f, "Cel:").pack(anchor="w", padx=6)
        self.var_goal = tk.StringVar(value="minimize")
        frame_goal = tk.Frame(f, bg="#1e1e2e")
        frame_goal.pack(fill=tk.X, padx=6)
        for txt, val in [("Minimalizacja", "minimize"), ("Maksymalizacja", "maximize")]:
            tk.Radiobutton(frame_goal, text=txt, variable=self.var_goal, value=val,
                           bg="#1e1e2e", fg="#cdd6f4", selectcolor="#313244",
                           activebackground="#1e1e2e").pack(side=tk.LEFT)

        # ── Populacja ─────────────────────────────────────────────────────────
        self._section(f, "Populacja")
        self._label(f, "Rozmiar populacji:").pack(anchor="w", padx=6)
        self.var_pop = tk.IntVar(value=50)
        ttk.Spinbox(f, from_=10, to=1000, textvariable=self.var_pop,
                    width=10).pack(padx=6, pady=2, fill=tk.X)

        self._label(f, "Liczba epok:").pack(anchor="w", padx=6)
        self.var_epochs = tk.IntVar(value=100)
        ttk.Spinbox(f, from_=10, to=10000, textvariable=self.var_epochs,
                    width=10).pack(padx=6, pady=2, fill=tk.X)

        self._label(f, "Dokładność (miejsca po przecinku):").pack(anchor="w", padx=6)
        self.var_prec = tk.IntVar(value=6)
        ttk.Spinbox(f, from_=1, to=10, textvariable=self.var_prec,
                    width=10).pack(padx=6, pady=2, fill=tk.X)

        # ── Selekcja ──────────────────────────────────────────────────────────
        self._section(f, "Selekcja")
        self.var_sel = tk.StringVar(value="tournament")
        ttk.Combobox(f, textvariable=self.var_sel,
                     values=list(SELECTION_METHODS.keys()), state="readonly",
                     width=26).pack(padx=6, pady=2, fill=tk.X)

        self._label(f, "Rozmiar turnieju:").pack(anchor="w", padx=6)
        self.var_tour = tk.IntVar(value=3)
        ttk.Spinbox(f, from_=2, to=20, textvariable=self.var_tour,
                    width=10).pack(padx=6, pady=2, fill=tk.X)

        # ── Krzyżowanie ───────────────────────────────────────────────────────
        self._section(f, "Krzyżowanie")
        self.var_cross = tk.StringVar(value="single_point")
        ttk.Combobox(f, textvariable=self.var_cross,
                     values=list(CROSSOVER_METHODS.keys()), state="readonly",
                     width=26).pack(padx=6, pady=2, fill=tk.X)
        self._label(f, "Prawdopodobieństwo krzyżowania:").pack(anchor="w", padx=6)
        self.var_cp = tk.DoubleVar(value=0.8)
        ttk.Spinbox(f, from_=0.0, to=1.0, increment=0.05,
                    textvariable=self.var_cp, width=10, format="%.2f").pack(padx=6, pady=2, fill=tk.X)

        # ── Mutacja ───────────────────────────────────────────────────────────
        self._section(f, "Mutacja")
        self.var_mut = tk.StringVar(value="single_point")
        ttk.Combobox(f, textvariable=self.var_mut,
                     values=list(MUTATION_METHODS.keys()), state="readonly",
                     width=26).pack(padx=6, pady=2, fill=tk.X)
        self._label(f, "Prawdopodobieństwo mutacji:").pack(anchor="w", padx=6)
        self.var_mp = tk.DoubleVar(value=0.01)
        ttk.Spinbox(f, from_=0.0, to=1.0, increment=0.005,
                    textvariable=self.var_mp, width=10, format="%.3f").pack(padx=6, pady=2, fill=tk.X)

        # ── Operatory dodatkowe ───────────────────────────────────────────────
        self._section(f, "Operatory dodatkowe")
        self._label(f, "Praw. inwersji:").pack(anchor="w", padx=6)
        self.var_inv = tk.DoubleVar(value=0.01)
        ttk.Spinbox(f, from_=0.0, to=1.0, increment=0.005,
                    textvariable=self.var_inv, width=10, format="%.3f").pack(padx=6, pady=2, fill=tk.X)
        self._label(f, "Liczba osobników elitarnych:").pack(anchor="w", padx=6)
        self.var_elite = tk.IntVar(value=1)
        ttk.Spinbox(f, from_=0, to=20, textvariable=self.var_elite,
                    width=10).pack(padx=6, pady=2, fill=tk.X)

        # ── Przyciski ─────────────────────────────────────────────────────────
        btn_frame = tk.Frame(f, bg="#1e1e2e")
        btn_frame.pack(fill=tk.X, pady=10, padx=6)

        self.btn_run = tk.Button(btn_frame, text="▶  Start",
                                 command=self._start,
                                 bg="#a6e3a1", fg="#1e1e2e",
                                 font=("Segoe UI", 10, "bold"),
                                 relief="flat", cursor="hand2")
        self.btn_run.pack(fill=tk.X, pady=2)

        self.btn_stop = tk.Button(btn_frame, text="⬛  Stop",
                                  command=self._stop,
                                  bg="#f38ba8", fg="#1e1e2e",
                                  font=("Segoe UI", 10, "bold"),
                                  relief="flat", cursor="hand2",
                                  state="disabled")
        self.btn_stop.pack(fill=tk.X, pady=2)

        self.btn_save = tk.Button(btn_frame, text="💾  Zapisz wyniki",
                                  command=self._save,
                                  bg="#89b4fa", fg="#1e1e2e",
                                  font=("Segoe UI", 10, "bold"),
                                  relief="flat", cursor="hand2",
                                  state="disabled")
        self.btn_save.pack(fill=tk.X, pady=2)

        # Status / czas / wynik
        self._section(f, "Status")
        self.lbl_status = self._label(f, "Gotowy.")
        self.lbl_status.pack(anchor="w", padx=6)
        self.lbl_time = self._label(f, "Czas: —")
        self.lbl_time.pack(anchor="w", padx=6)
        self.lbl_best = self._label(f, "Najlepszy wynik: —")
        self.lbl_best.pack(anchor="w", padx=6, pady=(0, 6))

        self.progress = ttk.Progressbar(f, mode="determinate")
        self.progress.pack(fill=tk.X, padx=6, pady=4)

    # ── Logika Start / Stop / Save ────────────────────────────────────────────

    def _build_config(self) -> GAConfig:
        n = self.var_n.get()
        func_cls = TEST_FUNCTIONS[self.var_func.get()]
        func_obj = func_cls(n_variables=n)
        return GAConfig(
            population_size=self.var_pop.get(),
            n_epochs=self.var_epochs.get(),
            n_variables=n,
            bounds=func_obj.bounds,
            precision=self.var_prec.get(),
            selection_method=self.var_sel.get(),
            crossover_method=self.var_cross.get(),
            mutation_method=self.var_mut.get(),
            crossover_prob=self.var_cp.get(),
            mutation_prob=self.var_mp.get(),
            inversion_prob=self.var_inv.get(),
            tournament_size=self.var_tour.get(),
            elite_size=self.var_elite.get(),
            maximize=self.var_goal.get() == "maximize",
        )

    def _start(self):
        self.plot_panel.clear()
        config = self._build_config()
        func_cls = TEST_FUNCTIONS[self.var_func.get()]
        func = func_cls(n_variables=config.n_variables)

        self._history_best = []
        self._history_avg = []
        self._config = config
        self._func_name = self.var_func.get()

        self.progress["maximum"] = config.n_epochs
        self.progress["value"] = 0
        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_save.config(state="disabled")
        self.lbl_status.config(text="Trwa obliczanie…")

        def callback(epoch, best, avg):
            self._history_best.append(best)
            self._history_avg.append(avg)
            self.after(0, self._update_progress, epoch, best, avg, config.n_epochs)

        self._ga = GeneticAlgorithm(func, config, progress_callback=callback)
        self._thread = threading.Thread(target=self._run_ga, daemon=True)
        self._thread.start()

    def _run_ga(self):
        result = self._ga.run()
        self._result = result
        self.after(0, self._on_done, result)

    def _stop(self):
        if self._ga:
            self._ga.stop()

    def _update_progress(self, epoch, best, avg, total):
        self.progress["value"] = epoch
        self.lbl_status.config(text=f"Epoka {epoch}/{total}")
        self.lbl_best.config(text=f"Najlepszy: {best:.6f}")
        self.plot_panel.update_plot(self._history_best, self._history_avg)

    def _on_done(self, result):
        self.btn_run.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_save.config(state="normal")
        self.lbl_status.config(text="Zakończono.")
        self.lbl_time.config(text=f"Czas: {result.elapsed_time:.3f} s")
        self.lbl_best.config(text=f"Najlepszy: {result.best_fitness:.8f}")
        self.plot_panel.update_plot(self._history_best, self._history_avg, final=True)

    def _save(self):
        if self._result is None:
            return
        out_dir = filedialog.askdirectory(title="Wybierz folder do zapisu")
        if not out_dir:
            return
        paths = save_results(self._result, self._config, self._func_name, out_dir)
        messagebox.showinfo("Zapisano",
                            f"Wyniki zapisane:\n{paths['csv']}\n{paths['json']}")


# ── Pomocniczy ScrollFrame ─────────────────────────────────────────────────────

class _ScrollFrame(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg="#1e1e2e", **kw)
        canvas = tk.Canvas(self, bg="#1e1e2e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = tk.Frame(canvas, bg="#1e1e2e")
        self.inner.bind("<Configure>",
                        lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.bind("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()

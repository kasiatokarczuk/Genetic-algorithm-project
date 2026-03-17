"""
Panel wykresów – historia zbieżności algorytmu.
"""
import tkinter as tk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class PlotPanel(tk.Frame):
    """
    Osadzony w GUI panel matplotlib pokazujący:
      - najlepszy fitness w każdej epoce
      - średni fitness w każdej epoce
    """

    def __init__(self, parent, **kw):
        super().__init__(parent, bg="#1e1e2e", **kw)
        self._build()

    def _build(self):
        self.fig = Figure(figsize=(6, 4), facecolor="#1e1e2e")
        self.ax = self.fig.add_subplot(111)
        self._style_ax()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

    def _style_ax(self):
        ax = self.ax
        ax.set_facecolor("#181825")
        ax.tick_params(colors="#cdd6f4", labelsize=8)
        ax.spines["bottom"].set_color("#45475a")
        ax.spines["left"].set_color("#45475a")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Epoka", color="#a6adc8", fontsize=9)
        ax.set_ylabel("Wartość funkcji", color="#a6adc8", fontsize=9)
        ax.set_title("Zbieżność algorytmu genetycznego",
                     color="#cba6f7", fontsize=11, pad=10)
        ax.grid(True, color="#313244", linewidth=0.5, linestyle="--")

    def clear(self):
        self.ax.cla()
        self._style_ax()
        self.canvas.draw()

    def update_plot(self, history_best: list, history_avg: list,
                    final: bool = False):
        ax = self.ax
        ax.cla()
        self._style_ax()

        epochs = range(1, len(history_best) + 1)

        if history_best:
            ax.plot(epochs, history_best, color="#a6e3a1", linewidth=1.5,
                    label="Najlepszy")
        if history_avg:
            ax.plot(epochs, history_avg, color="#89b4fa", linewidth=1.0,
                    linestyle="--", alpha=0.7, label="Średni")

        ax.legend(facecolor="#313244", edgecolor="#45475a",
                  labelcolor="#cdd6f4", fontsize=8)

        if final and history_best:
            best_epoch = history_best.index(min(history_best)) + 1
            ax.axvline(best_epoch, color="#f38ba8", linewidth=0.8,
                       linestyle=":", label=f"Min epoka {best_epoch}")
            ax.annotate(f"  min={min(history_best):.5f}",
                        xy=(best_epoch, min(history_best)),
                        xytext=(5, 10), textcoords="offset points",
                        color="#f38ba8", fontsize=7)

        self.canvas.draw()

# Projekt 1 – Algorytm Genetyczny (Obliczenia Ewolucyjne 2025/26)

## Funkcja testowa: Rosenbrock
```
f(x) = Σ_{i=0}^{N-2} [ 100·(x_{i+1} - x_i²)² + (x_i - 1)² ]
Zakres:  [-2.048, 2.048]^N
Minimum: f(1, 1, ..., 1) = 0.0
```

---

## Struktura projektu

```
genetic_algorithm/
│
├── core/                        ← OSOBA 1 (logika)
│   ├── chromosome.py            # Binarna reprezentacja, kodowanie/dekodowanie
│   ├── selection.py             # Selekcja: najlepszych, ruletki, turniejowa
│   ├── crossover.py             # Krzyżowanie: 1-pkt, 2-pkt, jednorodne, ziarniste
│   ├── mutation.py              # Mutacje: brzegowa, 1-pkt, 2-pkt
│   ├── operators.py             # Inwersja + funkcje testowe (Rosenbrock)
│   ├── genetic_algorithm.py     # Główna pętla GA + GAConfig + GAResult
│   └── results_io.py            # Zapis wyników do CSV i JSON
│
├── gui/                         ← OSOBA 2 (wizualizacja)
│   ├── main_window.py           # Główne okno tkinter z panelem konfiguracji
│   └── plot_panel.py            # Panel matplotlib (wykresy zbieżności)
│
├── tests/
│   └── test_ga.py               # Testy bez GUI (uruchamianie logiki wprost)
│
├── results/                     # Folder na wyniki (tworzony automatycznie)
├── main.py                      # Punkt startowy: python main.py
└── requirements.txt
```

---

## Podział pracy

### Osoba 1 – Logika (`core/`)
- Implementacja wszystkich operatorów GA
- Funkcja testowa Rosenbrock (N zmiennych)
- `GAConfig` – wszystkie parametry algorytmu
- `GAResult` – historia epok, najlepsze rozwiązanie, czas
- Zapis wyników (CSV + JSON)
- Testy headless w `tests/test_ga.py`

### Osoba 2 – GUI (`gui/`)
- Okno konfiguracji ze scrollowanym panelem
- Spinboxy / combobox dla wszystkich parametrów
- Pasek postępu + labels ze statusem i czasem
- Osadzony wykres matplotlib (zbieżność w czasie rzeczywistym)
- Przycisk zapisu wyników (z dialogiem wyboru folderu)

---

## Uruchomienie

```bash
pip install -r requirements.txt
python main.py           # uruchamia GUI
python tests/test_ga.py  # testy bez GUI
```

---

## Zaimplementowane elementy (wg wymagań)

| Element                          | Status |
|----------------------------------|--------|
| Binarna reprezentacja chromosomu | ✅     |
| Konfiguracja dokładności         | ✅     |
| Konfiguracja rozmiaru populacji  | ✅     |
| Konfiguracja liczby epok         | ✅     |
| Selekcja: najlepszych            | ✅     |
| Selekcja: ruletki                | ✅     |
| Selekcja: turniejowa             | ✅     |
| Krzyżowanie 1-punktowe           | ✅     |
| Krzyżowanie 2-punktowe           | ✅     |
| Krzyżowanie jednorodne           | ✅     |
| Krzyżowanie ziarniste            | ✅     |
| Mutacja brzegowa                 | ✅     |
| Mutacja 1-punktowa               | ✅     |
| Mutacja 2-punktowa               | ✅     |
| Operator inwersji                | ✅     |
| Strategia elitarna               | ✅     |
| Funkcja Rosenbrock (N zmiennych) | ✅     |
| GUI z konfiguracją               | ✅     |
| Wykresy zbieżności               | ✅     |
| Czas obliczeń                    | ✅     |
| Zapis wyników (CSV + JSON)       | ✅     |
| Minimalizacja i maksymalizacja   | ✅     |

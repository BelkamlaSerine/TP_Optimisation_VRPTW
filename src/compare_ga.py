"""
compare_ga.py : compare 8 variantes de l'algorithme genetique (selection x croisement x remplacement)
sur la grande instance Homberger C1_2_1 (200 clients). Affiche Best Solution et BKS a chaque execution.
"""
import csv
import itertools
import os
import statistics

from data_loader import load_homberger_instance
from heuristic_method import validate_solution
from metaheuristic import heuristic_initialization
from genetic_algorithm import genetic_algorithm

# ------------------- PARAMETRES -------------------
INSTANCE_FILE = "C1_2_1.TXT"
INSTANCE_NAME = "Homberger_200_C1_2_1"
BKS = 2704.57          # Best Known Solution (litterature / optimum CBC)
N_RUNS = 5             # graines 0..4
MAX_EVALS = 100000     # budget identique pour toutes les variantes
POP_SIZE = 100
TIME_LIMIT = None      # ex: 60 secondes par run, ou None
# --------------------------------------------------

SELECTIONS = ["elitist", "tournament"]
CROSSOVERS = ["one_point", "two_point"]
REPLACEMENTS = ["elitist", "steady_state"]

instance = load_homberger_instance(INSTANCE_FILE)
init = heuristic_initialization(instance, "clarke_wright")

rows, histories = [], {}

for sel, cx, rep in itertools.product(SELECTIONS, CROSSOVERS, REPLACEMENTS):
    name = f"{sel[:4]}+{cx}+{rep}"
    for run in range(N_RUNS):
        res = genetic_algorithm(
            instance, init,
            selection=sel, crossover=cx, replacement=rep,
            pop_size=POP_SIZE, max_evals=MAX_EVALS,
            seed=run, time_limit=TIME_LIMIT,
        )
        val = validate_solution(instance, res)
        gap = (res["distance"] - BKS) / BKS * 100

        rows.append({
            "Instance": INSTANCE_NAME, "Variant": name,
            "Selection": sel, "Crossover": cx, "Replacement": rep,
            "Run": run + 1,
            "Best_Solution": round(res["distance"], 2),
            "BKS": BKS,
            "Gap_%": round(gap, 2),
            "Vehicles": res["vehicles"],
            "Time": round(res["execution_time"], 2),
            "Feasible": val["solution_feasible"],
        })
        if run == 0:
            histories[name] = res["history"]

        print(f"{name:34s} run {run + 1}: Best Solution = {res['distance']:9.2f} "
              f"| BKS = {BKS} | gap = {gap:6.2f}% | {res['vehicles']} veh "
              f"| feasible={val['solution_feasible']}")

# ---------------- CSV ----------------
os.makedirs("results", exist_ok=True)
with open("results/ga_comparison.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ---------------- Tableau recapitulatif ----------------
names = list(histories.keys())
print("\n" + "=" * 100)
print(f"{'Variante':36s}{'Moy.':>10s}{'Best':>10s}{'Std':>8s}{'Veh':>6s}"
      f"{'Gap moy%':>10s}{'Gap best%':>11s}{'t(s)':>7s}")
for name in names:
    sub = [r for r in rows if r["Variant"] == name]
    d = [r["Best_Solution"] for r in sub]
    print(f"{name:36s}{statistics.mean(d):10.2f}{min(d):10.2f}"
          f"{(statistics.stdev(d) if len(d) > 1 else 0):8.2f}"
          f"{statistics.mean(r['Vehicles'] for r in sub):6.1f}"
          f"{(statistics.mean(d) - BKS) / BKS * 100:10.2f}"
          f"{(min(d) - BKS) / BKS * 100:11.2f}"
          f"{statistics.mean(r['Time'] for r in sub):7.1f}")
print("Saved: results/ga_comparison.csv")

# ---------------- Courbes ----------------
try:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for name, hist in histories.items():
        ax1.plot([h[0] for h in hist], [h[1] for h in hist], label=name)
    ax1.axhline(BKS, color="green", ls="--", label=f"BKS {BKS}")
    ax1.set_xlabel("Evaluations")
    ax1.set_ylabel("Best distance")
    ax1.set_title("Convergence des variantes GA (seed 0)")
    ax1.legend(fontsize=7)
    ax1.grid(True)

    means = [statistics.mean(r["Best_Solution"] for r in rows if r["Variant"] == nm) for nm in names]
    ax2.bar(range(len(names)), means)
    ax2.axhline(BKS, color="green", ls="--", label=f"BKS {BKS}")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax2.set_ylim(BKS * 0.97, max(means) * 1.02)
    ax2.set_ylabel("Distance moyenne (5 runs)")
    ax2.set_title("Comparaison des variantes")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("results/ga_comparison.png", dpi=150)
    print("Plot saved in: results/ga_comparison.png")
except ImportError:
    print("matplotlib non installe : pip install matplotlib")

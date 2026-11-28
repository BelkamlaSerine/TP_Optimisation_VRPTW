import csv, os, statistics

from data_loader import load_homberger_instance
from heuristic_method import validate_solution
from metaheuristic import heuristic_initialization, simulated_annealing
from metaheuristic_v2 import simulated_annealing_v2

INSTANCE_FILE = "C1_2_1.TXT"
OPTIMAL = 2704.57
N_RUNS = 5            # memes graines 0..4 que la v1
ITERATIONS = 500000

instance = load_homberger_instance(INSTANCE_FILE)
init = heuristic_initialization(instance, "clarke_wright")

# (nom, fonction, parametres)  -> chaque ligne ajoute UNE amelioration
CONFIGS = [
    ("V1 (SA original)",            simulated_annealing,    {}),
    ("V2a voisinage guide + Or-opt/2-opt*", simulated_annealing_v2, dict(use_elimination=False)),
    ("V2b + elimination de tournee", simulated_annealing_v2, dict()),
    ("V2c + lam=50",                 simulated_annealing_v2, dict(lam=50)),
    ("V2d + reheating (4 cycles)",   simulated_annealing_v2, dict(lam=50, n_cycles=4)),
    ("V2e + T0=50",                  simulated_annealing_v2, dict(lam=50, n_cycles=4, t_start=50)),
]

rows = []
for name, fn, kw in CONFIGS:
    for run in range(N_RUNS):
        res = fn(instance, init, iterations=ITERATIONS, seed=run, **kw)
        val = validate_solution(instance, res)
        rows.append(dict(Config=name, Run=run + 1, Distance=round(res["distance"], 2),
                         Vehicles=res["vehicles"], Time=round(res["execution_time"], 2),
                         Feasible=val["solution_feasible"]))
        print(f"{name:38s} run {run+1}: {res['distance']:9.2f} ({res['vehicles']} veh) feasible={val['solution_feasible']}")

os.makedirs("results", exist_ok=True)
with open("results/improvement_comparison.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print("\n" + "=" * 100)
print(f"{'Config':40s}{'Moy.':>10s}{'Best':>10s}{'Std':>8s}{'Veh':>6s}{'Gap moy%':>10s}{'Gap best%':>11s}{'t(s)':>7s}")
for name, _, _ in CONFIGS:
    sub = [r for r in rows if r["Config"] == name]
    d = [r["Distance"] for r in sub]
    print(f"{name:40s}{statistics.mean(d):10.2f}{min(d):10.2f}{(statistics.stdev(d) if len(d)>1 else 0):8.2f}"
          f"{statistics.mean(r['Vehicles'] for r in sub):6.1f}"
          f"{(statistics.mean(d)-OPTIMAL)/OPTIMAL*100:10.2f}{(min(d)-OPTIMAL)/OPTIMAL*100:11.2f}"
          f"{statistics.mean(r['Time'] for r in sub):7.1f}")
print("Saved: results/improvement_comparison.csv")

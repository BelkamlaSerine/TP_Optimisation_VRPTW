import csv
import os
import statistics

from data_loader import load_homberger_instance
from heuristic_method import calculate_route_distance, validate_solution
from metaheuristic import (
    heuristic_initialization,
    random_initialization,
    simulated_annealing,
)

# ------------------- PARAMETRES -------------------
INSTANCE_FILE = "C1_2_1.TXT"
INSTANCE_NAME = "Homberger_200_C1_2_1"
N_RUNS = 5            # nombre d'executions (seeds differentes)
ITERATIONS = 500000
TIME_LIMIT = None     # ex: 60 secondes par run, ou None
# --------------------------------------------------

instance = load_homberger_instance(INSTANCE_FILE)
customers = {c["id"]: c for c in instance["customers"]}
depot = instance["depot"]


def total_distance(routes):
    return sum(calculate_route_distance(r, customers, depot) for r in routes)


rows = []
histories = {"Random": None, "Heuristic": None}

for init_type in ["Random", "Heuristic"]:
    for run in range(N_RUNS):

        seed = run

        if init_type == "Random":
            init_routes = random_initialization(instance, seed=seed)
        else:
            init_routes = heuristic_initialization(instance, "clarke_wright")

        init_dist = total_distance(init_routes)
        init_veh = len(init_routes)

        result = simulated_annealing(
            instance,
            init_routes,
            iterations=ITERATIONS,
            seed=seed,
            time_limit=TIME_LIMIT,
        )

        validation = validate_solution(instance, result)

        rows.append({
            "Instance": INSTANCE_NAME,
            "Init": init_type,
            "Run": run + 1,
            "Init_Distance": round(init_dist, 2),
            "Init_Vehicles": init_veh,
            "Final_Distance": round(result["distance"], 2),
            "Final_Vehicles": result["vehicles"],
            "Time": round(result["execution_time"], 2),
            "Feasible": validation["solution_feasible"],
        })

        if histories[init_type] is None:
            histories[init_type] = result["history"]

        print(
            f"{init_type:9s} run {run + 1}: "
            f"init {init_dist:9.2f} ({init_veh} veh) -> "
            f"final {result['distance']:9.2f} ({result['vehicles']} veh) "
            f"| feasible={validation['solution_feasible']}"
        )

# ---------------- Sauvegarde CSV ----------------
os.makedirs("results", exist_ok=True)
csv_path = "results/init_comparison.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

# ---------------- Tableau recapitulatif ----------------
print("\n========== SUMMARY ==========")
print(f"{'Init':10s}{'Init dist':>12s}{'Final mean':>12s}{'Final best':>12s}"
      f"{'Std':>9s}{'Veh (mean)':>12s}{'Time (mean)':>13s}")

for init_type in ["Random", "Heuristic"]:
    sub = [r for r in rows if r["Init"] == init_type]
    finals = [r["Final_Distance"] for r in sub]

    print(
        f"{init_type:10s}"
        f"{statistics.mean(r['Init_Distance'] for r in sub):12.2f}"
        f"{statistics.mean(finals):12.2f}"
        f"{min(finals):12.2f}"
        f"{(statistics.stdev(finals) if len(finals) > 1 else 0):9.2f}"
        f"{statistics.mean(r['Final_Vehicles'] for r in sub):12.1f}"
        f"{statistics.mean(r['Time'] for r in sub):13.2f}"
    )

print("Results saved in:", csv_path)

# ---------------- Courbe de convergence ----------------
try:
    import matplotlib.pyplot as plt

    for init_type, hist in histories.items():
        plt.plot([h[0] for h in hist], [h[1] for h in hist], label=init_type)

    plt.xlabel("Iterations")
    plt.ylabel("Best distance")
    plt.title("Random vs Heuristic initialization")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/init_convergence.png", dpi=150)
    print("Plot saved in: results/init_convergence.png")
except ImportError:
    print("matplotlib non installe : pip install matplotlib")

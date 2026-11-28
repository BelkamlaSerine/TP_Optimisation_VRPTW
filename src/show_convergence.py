"""Affiche (pendant l'execution) la courbe de convergence V1 vs V2 + les tournees de la meilleure solution."""
import matplotlib.pyplot as plt

from data_loader import load_homberger_instance
from heuristic_method import validate_solution
from metaheuristic import heuristic_initialization, simulated_annealing
from metaheuristic_v2 import simulated_annealing_v2

INSTANCE_FILE = "C1_2_1.TXT"
OPTIMAL = 2704.57
ITERATIONS = 500000
SEED = 0

instance = load_homberger_instance(INSTANCE_FILE)
depot = instance["depot"]
cust = {c["id"]: c for c in instance["customers"]}
init = heuristic_initialization(instance, "clarke_wright")

print("Execution V1 ...")
r1 = simulated_annealing(instance, init, iterations=ITERATIONS, seed=SEED)
print(f"V1 : {r1['distance']:.2f} ({r1['vehicles']} veh)")

print("Execution V2 ...")
r2 = simulated_annealing_v2(instance, init, iterations=ITERATIONS, seed=SEED)
print(f"V2 : {r2['distance']:.2f} ({r2['vehicles']} veh) feasible="
      f"{validate_solution(instance, r2)['solution_feasible']}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- courbe de convergence
ax1.plot([h[0] for h in r1["history"]], [h[1] for h in r1["history"]],
         label=f"V1 (SA original) : {r1['distance']:.1f}")
ax1.plot([h[0] for h in r2["history"]], [h[1] for h in r2["history"]],
         label=f"V2 (SA ameliore) : {r2['distance']:.1f}")
ax1.axhline(OPTIMAL, color="green", ls="--", label=f"Optimum CBC : {OPTIMAL}")
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Meilleure distance")
ax1.set_title("Convergence : V1 vs V2 (C1_2_1, 200 clients)")
ax1.legend()
ax1.grid(True)

# --- tournees de la solution V2
ax2.scatter([c["x"] for c in cust.values()], [c["y"] for c in cust.values()], s=8, c="gray")
for route in r2["routes"]:
    pts = [(depot["x"], depot["y"])] + [(cust[c]["x"], cust[c]["y"]) for c in route] + [(depot["x"], depot["y"])]
    ax2.plot([p[0] for p in pts], [p[1] for p in pts], lw=1)
ax2.scatter([depot["x"]], [depot["y"]], marker="s", s=80, c="red", zorder=5, label="Depot")
ax2.set_title(f"Tournees V2 : {r2['vehicles']} vehicules, distance {r2['distance']:.1f}")
ax2.legend()
ax2.set_aspect("equal")

plt.tight_layout()
plt.savefig("results/convergence_v1_v2.png", dpi=150)
plt.show()


import math
import time
import pulp
import csv
import os


def solve_exact(instance, time_limit=300, instance_name="Unknown"):

    customers = instance["customers"]
    depot = instance["depot"]

    n = len(customers)
    Q = instance["capacity"]
    max_vehicles = instance["vehicles"]

    ids = [c["id"] for c in customers]

    start = 0
    end = n + 1

    # Depot de retour : meme position que le depot initial
    nodes = {c["id"]: c for c in customers}

    def distance(i, j):
        a = depot if i == start else nodes[i]
        b = depot if j == end else nodes[j]

        return math.hypot(
            a["x"] - b["x"],
            a["y"] - b["y"]
        )

    # Distances et temps de trajet
    d = {}

    arcs = (
        [(start, j) for j in ids]
        + [(i, j) for i in ids for j in ids if i != j]
        + [(i, end) for i in ids]
    )

    for i, j in arcs:
        d[i, j] = distance(i, j)

    # Modele MILP
    model = pulp.LpProblem(
        "VRPTW_Exact",
        pulp.LpMinimize
    )

    # Variables binaires
    x = pulp.LpVariable.dicts(
        "x", arcs, cat=pulp.LpBinary
    )

    # Variables de charge
    u = pulp.LpVariable.dicts(
        "load", ids, lowBound=0, upBound=Q
    )

    # Variables temporelles
    w = pulp.LpVariable.dicts(
        "time", ids, lowBound=0
    )

    # Fonction objectif : distance totale
    model += pulp.lpSum(
        d[i, j] * x[i, j] for i, j in arcs
    )

    # Chaque client quitte exactement une fois
    for i in ids:
        model += pulp.lpSum(
            x[i, j] for j in ids + [end]
            if (i, j) in x
        ) == 1

    # Chaque client est visite exactement une fois
    for j in ids:
        model += pulp.lpSum(
            x[i, j] for i in [start] + ids
            if (i, j) in x
        ) == 1

    # Nombre maximal de vehicules
    model += pulp.lpSum(
        x[start, j] for j in ids
    ) <= max_vehicles

    # Contraintes de capacite
    for i in ids:
        qi = nodes[i]["demand"]
        model += u[i] >= qi
        model += u[i] <= Q

    for i in ids:
        for j in ids:
            if i != j:
                qj = nodes[j]["demand"]

                model += (
                    u[j] >= u[i] + qj
                    - Q * (1 - x[i, j])
                )

    # Fenetres temporelles
    for i in ids:
        model += w[i] >= nodes[i]["ready_time"]
        model += w[i] <= nodes[i]["due_time"]

    # Temps de trajet : convention benchmark t_ij = d_ij
    for j in ids:
        M = max(
            0,
            depot["due_time"] + d[start, j]
            - nodes[j]["ready_time"]
        )

        model += (
            w[j] >= d[start, j]
            - M * (1 - x[start, j])
        )

    for i in ids:
        for j in ids:
            if i != j:
                si = nodes[i]["service_time"]

                M = max(
                    0,
                    nodes[i]["due_time"] + si + d[i, j]
                    - nodes[j]["ready_time"]
                )

                model += (
                    w[j] >= w[i] + si + d[i, j]
                    - M * (1 - x[i, j])
                )

    # Retour au depot avant sa fermeture
    for i in ids:
        si = nodes[i]["service_time"]

        M = max(
            0,
            nodes[i]["due_time"] + si + d[i, end]
            - depot["due_time"]
        )

        model += (
            w[i] + si + d[i, end]
            <= depot["due_time"] + M * (1 - x[i, end])
        )

    # Resolution exacte avec CBC
    solver = pulp.PULP_CBC_CMD(
        msg=True,
        timeLimit=time_limit
    )

    start_time = time.perf_counter()
    model.solve(solver)
    execution_time = time.perf_counter() - start_time

    status = pulp.LpStatus[model.status]

    print("\n========== EXACT METHOD ==========")
    print("Instance:", instance_name)
    print("Status:", status)
    print("Execution time:", round(execution_time, 2), "seconds")

    if status != "Optimal":
        print("Optimal solution not proven.")
        print("No results saved for this run.")
        return None

    total_distance = pulp.value(model.objective)

    print("Total distance:", round(total_distance, 2))

    # Reconstruction des tournees
    routes = []
    used = 0

    for j in ids:
        if pulp.value(x[start, j]) > 0.5:
            used += 1
            route = [start, j]
            current = j

            while current != end:
                next_node = None

                for k in ids + [end]:
                    if (current, k) in x:
                        if pulp.value(x[current, k]) > 0.5:
                            next_node = k
                            break

                if next_node is None:
                    break

                route.append(next_node)
                current = next_node

            routes.append(route)

    print("Vehicles used:", used)

    for index, route in enumerate(routes, 1):
        print(f"Vehicle {index}: {route}")

    print("==================================")

    # Sauvegarde des resultats dans un fichier CSV
    os.makedirs("results", exist_ok=True)

    csv_file = os.path.join(
        "results",
        "exact_results.csv"
    )

    gap = 0.0 if status == "Optimal" else ""

    file_exists = os.path.isfile(csv_file)

    with open(
        csv_file,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Instance",
                "Status",
                "Distance",
                "Execution_Time",
                "Vehicles_Used",
                "Gap",
                "Routes"
            ]
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "Instance": instance_name,
            "Status": status,
            "Distance": round(total_distance, 2),
            "Execution_Time": round(execution_time, 2),
            "Vehicles_Used": used,
            "Gap": gap,
            "Routes": str(routes)
        })

    print("Results saved in:", csv_file)

    return {
        "instance": instance_name,
        "status": status,
        "distance": total_distance,
        "execution_time": execution_time,
        "vehicles_used": used,
        "gap": gap,
        "routes": routes
    }

import math
import random
import time

from heuristic_method import (
    calculate_route_distance,
    check_route_feasibility,
    clarke_wright,
    solomon_insertion,
)


# ---------------------------------------------------------------
# 1) INITIALISATIONS
# ---------------------------------------------------------------

def random_initialization(instance, seed=None):
    """
    Initialisation aleatoire :
    on melange les clients, puis on les ajoute dans l'ordre a la tournee
    courante tant que c'est faisable (capacite + fenetres de temps).
    Sinon on ouvre une nouvelle tournee.
    """
    rng = random.Random(seed)

    depot = instance["depot"]
    capacity = instance["capacity"]
    customers = {c["id"]: c for c in instance["customers"]}

    ids = list(customers.keys())
    rng.shuffle(ids)

    routes = []
    current = []

    for cid in ids:
        candidate = current + [cid]

        if check_route_feasibility(candidate, customers, depot, capacity):
            current = candidate
        else:
            if current:
                routes.append(current)
            current = [cid]

    if current:
        routes.append(current)

    return routes


def heuristic_initialization(instance, method="clarke_wright"):
    """
    Initialisation par heuristique constructive.
    Clarke & Wright est rapide ; Solomon est tres lent pour 200 clients.
    """
    if method == "solomon":
        result = solomon_insertion(instance)
    else:
        result = clarke_wright(instance)

    return [route[:] for route in result["routes"]]


# ---------------------------------------------------------------
# 2) RECUIT SIMULE (amelioration a partir d'une solution initiale)
# ---------------------------------------------------------------

def simulated_annealing(
    instance,
    init_routes,
    iterations=30000,
    t_start=20.0,
    t_end=0.1,
    seed=0,
    time_limit=None,
):
    rng = random.Random(seed)

    depot = instance["depot"]
    capacity = instance["capacity"]
    customers = {c["id"]: c for c in instance["customers"]}

    def dist(route):
        return calculate_route_distance(route, customers, depot)

    def feasible(route):
        return (not route) or check_route_feasibility(
            route, customers, depot, capacity
        )

    routes = [r[:] for r in init_routes]
    dists = [dist(r) for r in routes]

    current = sum(dists)
    best = current
    best_routes = [r[:] for r in routes]

    history = [(0, best)]
    cooling = (t_end / t_start) ** (1.0 / iterations)
    temperature = t_start
    start = time.time()

    for it in range(1, iterations + 1):

        if time_limit and time.time() - start > time_limit:
            break

        temperature *= cooling

        a = rng.randrange(len(routes))
        ra = routes[a]

        # changes : liste de (index_route, nouvelle_route) ; index None = nouvelle tournee
        changes = None

        if rng.random() < 0.5:
            # ---- RELOCATE ----
            pos = rng.randrange(len(ra))
            cid = ra[pos]
            new_a = ra[:pos] + ra[pos + 1:]

            r = rng.random()

            if r < 0.05:
                # sortir le client dans une nouvelle tournee
                if len(ra) == 1:
                    continue
                changes = [(a, new_a), (None, [cid])]

            else:
                b = rng.randrange(len(routes))

                if b == a:
                    q = rng.randrange(len(new_a) + 1)
                    moved = new_a[:q] + [cid] + new_a[q:]
                    changes = [(a, moved)]
                else:
                    rb = routes[b]
                    q = rng.randrange(len(rb) + 1)
                    new_b = rb[:q] + [cid] + rb[q:]
                    changes = [(a, new_a), (b, new_b)]

        else:
            # ---- SWAP ----
            b = rng.randrange(len(routes))

            if b == a:
                if len(ra) < 2:
                    continue
                i, j = rng.sample(range(len(ra)), 2)
                new_a = ra[:]
                new_a[i], new_a[j] = new_a[j], new_a[i]
                changes = [(a, new_a)]
            else:
                rb = routes[b]
                i = rng.randrange(len(ra))
                j = rng.randrange(len(rb))
                new_a = ra[:]
                new_b = rb[:]
                new_a[i], new_b[j] = rb[j], ra[i]
                changes = [(a, new_a), (b, new_b)]

        # faisabilite
        if not all(feasible(route) for _, route in changes):
            continue

        # variation de cout
        delta = 0.0
        new_dists = []

        for idx, route in changes:
            nd = dist(route)
            new_dists.append(nd)
            delta += nd - (dists[idx] if idx is not None else 0.0)

        # critere d'acceptation (Metropolis)
        if delta < 0 or rng.random() < math.exp(-delta / temperature):

            for (idx, route), nd in zip(changes, new_dists):
                if idx is None:
                    routes.append(route)
                    dists.append(nd)
                else:
                    routes[idx] = route
                    dists[idx] = nd

            # supprimer les tournees vides
            keep = [k for k, r in enumerate(routes) if r]
            if len(keep) != len(routes):
                routes = [routes[k] for k in keep]
                dists = [dists[k] for k in keep]

            current += delta

            if current < best - 1e-9:
                best = current
                best_routes = [r[:] for r in routes]

        if it % 500 == 0:
            history.append((it, best))

    execution_time = time.time() - start

    return {
        "routes": best_routes,
        "distance": sum(dist(r) for r in best_routes),
        "vehicles": len(best_routes),
        "execution_time": execution_time,
        "history": history,
    }

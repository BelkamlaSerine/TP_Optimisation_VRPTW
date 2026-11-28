"""
genetic_algorithm.py : algorithme genetique pour VRPTW (meme API que metaheuristic.py)

Individu   : permutation des clients. Decodage glouton : on ajoute les clients dans l'ordre
             a la tournee courante tant que capacite + fenetres horaires + retour depot sont OK,
             sinon on ouvre une nouvelle tournee.
Fitness    : distance totale + lam * nb_tournees (a minimiser).

Variantes (les 3 axes demandes par l'encadrante) :
  selection   : "elitist"      -> parents tires parmi la meilleure moitie de la population
                "tournament"   -> tournoi de taille k
  crossover   : "one_point"    -> croisement a 1 point (ordre conserve, pas de doublons)
                "two_point"    -> croisement a 2 points (segment de P1 + reste dans l'ordre de P2)
  replacement : "elitist"      -> (mu+lambda) : parents + enfants, on garde les pop_size meilleurs
                "steady_state" -> 1 enfant par iteration, il remplace le pire s'il est meilleur
"""
import math
import random
import time


def genetic_algorithm(
    instance,
    init_routes=None,          # solution Clarke & Wright (liste de tournees) ou None
    selection="tournament",
    crossover="two_point",
    replacement="elitist",
    pop_size=100,
    max_evals=100000,          # budget identique pour toutes les variantes
    p_cross=0.9,
    p_mut=0.2,
    tournament_size=3,
    lam=50.0,
    seed=0,
    time_limit=None,
):
    rng = random.Random(seed)

    depot = instance["depot"]
    cap = instance["capacity"]
    cust = {c["id"]: c for c in instance["customers"]}
    ids = list(cust.keys())
    n = len(ids)

    # ---------- donnees precalculees ----------
    nodes = {0: depot, **cust}
    size = max(nodes) + 1
    D = [[0.0] * size for _ in range(size)]
    for i, a in nodes.items():
        for j, b in nodes.items():
            D[i][j] = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
    ready, due, svc, dem = [0] * size, [0] * size, [0] * size, [0] * size
    for i, a in nodes.items():
        ready[i], due[i], svc[i], dem[i] = (
            a["ready_time"], a["due_time"], a["service_time"], a["demand"])
    t0, due0 = depot["ready_time"], depot["due_time"]

    # ---------- decodage + fitness ----------
    def decode(perm):
        """Retourne (fitness, distance, routes)."""
        routes, cur = [], []
        total, d, t, prev, load = 0.0, 0.0, t0, 0, 0
        for c in perm:
            if cur:
                a = t + D[prev][c]
                if a < ready[c]:
                    a = ready[c]
                if (a <= due[c] and load + dem[c] <= cap
                        and a + svc[c] + D[c][0] <= due0):
                    d += D[prev][c]
                    load += dem[c]
                    t = a + svc[c]
                    prev = c
                    cur.append(c)
                    continue
                total += d + D[prev][0]
                routes.append(cur)
            # nouvelle tournee avec le client c
            a = t0 + D[0][c]
            if a < ready[c]:
                a = ready[c]
            cur = [c]
            d, load, t, prev = D[0][c], dem[c], a + svc[c], c
        if cur:
            total += d + D[prev][0]
            routes.append(cur)
        return total + lam * len(routes), total, routes

    # ---------- evaluation avec compteur ----------
    state = {"evals": 0, "best_fit": float("inf"), "best_perm": None, "best_dist": None}
    history = []
    next_mark = [0]

    def evaluate(perm):
        fit, dist, _ = decode(perm)
        state["evals"] += 1
        if fit < state["best_fit"] - 1e-9:
            state["best_fit"], state["best_perm"], state["best_dist"] = fit, perm[:], dist
        if state["evals"] >= next_mark[0]:
            history.append((state["evals"], state["best_dist"]))
            next_mark[0] += 1000
        return fit

    # ---------- population initiale ----------
    pop = []
    if init_routes:
        base = [c for r in init_routes for c in r]
        pop.append((evaluate(base), base))
        while len(pop) < pop_size // 2:
            # variante de la solution heuristique : ordre des tournees melange + quelques swaps
            rr = [r[:] for r in init_routes]
            rng.shuffle(rr)
            p = [c for r in rr for c in r]
            for _ in range(rng.randrange(0, 4)):
                i, j = rng.sample(range(n), 2)
                p[i], p[j] = p[j], p[i]
            pop.append((evaluate(p), p))
    while len(pop) < pop_size:
        p = ids[:]
        rng.shuffle(p)
        pop.append((evaluate(p), p))

    # ---------- operateurs ----------
    def select_parent(sorted_pop):
        if selection == "elitist":
            return sorted_pop[rng.randrange(max(2, len(sorted_pop) // 2))][1]
        cand = [sorted_pop[rng.randrange(len(sorted_pop))] for _ in range(tournament_size)]
        return min(cand, key=lambda x: x[0])[1]

    def cx_one_point(p1, p2):
        cut = rng.randrange(1, n)
        head = p1[:cut]
        s = set(head)
        return head + [g for g in p2 if g not in s]

    def cx_two_point(p1, p2):
        a, b = sorted(rng.sample(range(n), 2))
        b += 1
        seg = p1[a:b]
        s = set(seg)
        rest = [g for g in p2 if g not in s]
        return rest[:a] + seg + rest[a:]

    cx = cx_one_point if crossover == "one_point" else cx_two_point

    def mutate(p):
        if rng.random() < p_mut:
            i, j = sorted(rng.sample(range(n), 2))
            if rng.random() < 0.5:
                p[i], p[j] = p[j], p[i]                 # swap
            else:
                p[i:j + 1] = p[i:j + 1][::-1]           # inversion
        return p

    def make_child(sorted_pop):
        p1, p2 = select_parent(sorted_pop), select_parent(sorted_pop)
        child = cx(p1, p2) if rng.random() < p_cross else p1[:]
        return mutate(child)

    # ---------- boucle principale ----------
    start = time.time()

    def out_of_time():
        return time_limit and time.time() - start > time_limit

    pop.sort(key=lambda x: x[0])

    if replacement == "steady_state":
        while state["evals"] < max_evals and not out_of_time():
            child = make_child(pop)
            f = evaluate(child)
            if f < pop[-1][0]:
                pop[-1] = (f, child)
                pop.sort(key=lambda x: x[0])
    else:  # (mu + lambda) elitiste
        while state["evals"] < max_evals and not out_of_time():
            offspring = []
            for _ in range(pop_size):
                child = make_child(pop)
                offspring.append((evaluate(child), child))
            merged, seen = [], set()
            for f, p in sorted(pop + offspring, key=lambda x: x[0]):
                key = tuple(p)
                if key not in seen:                      # evite les doublons exacts
                    seen.add(key)
                    merged.append((f, p))
            pop = merged[:pop_size]
            while len(pop) < pop_size:                   # securite : completer si trop de doublons
                p = ids[:]
                rng.shuffle(p)
                pop.append((evaluate(p), p))
            pop.sort(key=lambda x: x[0])

    execution_time = time.time() - start
    _, dist, routes = decode(state["best_perm"])
    history.append((state["evals"], dist))
    return {
        "routes": routes,
        "distance": dist,
        "vehicles": len(routes),
        "execution_time": execution_time,
        "history": history,
    }

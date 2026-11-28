"""
metaheuristic_v2.py : recuit simule ameliore pour VRPTW (meme API que metaheuristic.py)

Ameliorations par rapport a la version 1 (chaque bloc est activable/desactivable) :
  A. Voisinage guide : le client est deplace pres d'un de ses k plus proches voisins
     (au lieu d'une tournee choisie au hasard) -> beaucoup plus de mouvements faisables.
  B. Nouveaux mouvements : Or-opt (segments de 2-3 clients), 2-opt* (echange de queues
     entre deux tournees), 2-opt intra-tournee.
  C. Elimination de tournee : on essaie de vider la plus petite tournee.
  D. Cout = distance + lam * nb_tournees (lam=0 -> distance seule, comme la v1).
  E. Rechauffage : n_cycles cycles de refroidissement, chacun repart de la meilleure solution.
  F. Matrice de distances precalculee + evaluation faisabilite/distance en un seul passage.
"""
import math
import random
import time

from metaheuristic import random_initialization, heuristic_initialization  # noqa: F401


def simulated_annealing_v2(
    instance,
    init_routes,
    iterations=500000,
    t_start=20.0,
    t_end=0.1,
    seed=0,
    time_limit=None,
    lam=0.0,
    n_cycles=1,
    k_neighbors=15,
    use_guided=True,
    use_oropt=True,
    use_2opt_star=True,
    use_elimination=True,
):
    rng = random.Random(seed)

    depot = instance["depot"]
    cap = instance["capacity"]
    max_veh = instance["vehicles"]
    cust = {c["id"]: c for c in instance["customers"]}
    ids = list(cust.keys())

    # ---------- F. donnees precalculees ----------
    nodes = {0: depot, **cust}
    size = max(nodes) + 1
    D = [[0.0] * size for _ in range(size)]
    for i, a in nodes.items():
        for j, b in nodes.items():
            D[i][j] = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
    ready = [0] * size
    due = [0] * size
    svc = [0] * size
    dem = [0] * size
    for i, a in nodes.items():
        ready[i], due[i], svc[i], dem[i] = (
            a["ready_time"], a["due_time"], a["service_time"], a["demand"])
    t0, due0 = depot["ready_time"], depot["due_time"]

    def ev(route):
        """Distance de la tournee, ou None si infaisable (capacite / TW / retour)."""
        if not route:
            return 0.0
        t, prev, load, d = t0, 0, 0, 0.0
        for c in route:
            dd = D[prev][c]
            a = t + dd
            if a < ready[c]:
                a = ready[c]
            if a > due[c]:
                return None
            load += dem[c]
            if load > cap:
                return None
            d += dd
            t = a + svc[c]
            prev = c
        back = D[prev][0]
        if t + back > due0:
            return None
        return d + back

    # ---------- A. voisins les plus proches ----------
    neigh = {c: sorted((o for o in ids if o != c), key=lambda o: D[c][o])[:k_neighbors]
             for c in ids}

    routes = [r[:] for r in init_routes]
    dists = [ev(r) for r in routes]
    route_of = {}

    def rebuild_map():
        route_of.clear()
        for k, r in enumerate(routes):
            for c in r:
                route_of[c] = k

    rebuild_map()

    def cost_of(ds, nr):
        return sum(ds) + lam * nr

    current = cost_of(dists, len(routes))
    best = current
    best_routes = [r[:] for r in routes]
    history = [(0, sum(dists))]

    start = time.time()
    cyc_iters = max(1, iterations // n_cycles)
    cooling = (t_end / t_start) ** (1.0 / cyc_iters)
    it_global = 0
    stop = False

    for cyc in range(n_cycles):
        if stop:
            break
        if cyc > 0:  # E. rechauffage : on repart de la meilleure solution
            routes = [r[:] for r in best_routes]
            dists = [ev(r) for r in routes]
            rebuild_map()
            current = cost_of(dists, len(routes))
        temperature = t_start

        for _ in range(cyc_iters):
            it_global += 1
            it = it_global
            if time_limit and (it & 255) == 0 and time.time() - start > time_limit:
                stop = True
                break
            temperature *= cooling

            changes = None
            r = rng.random()

            # ---------------- choix du mouvement ----------------
            if use_elimination and r < 0.004:
                changes = _eliminate(routes, dists, ev, rng)
            else:
                a = rng.randrange(len(routes))
                ra = routes[a]
                pos = rng.randrange(len(ra))
                cid = ra[pos]

                # voisin cible (guide) ou route aleatoire
                if use_guided and rng.random() < 0.9:
                    nb = neigh[cid][rng.randrange(len(neigh[cid]))]
                    b = route_of[nb]
                else:
                    nb = None
                    b = rng.randrange(len(routes))

                m = rng.random()

                if m < 0.30:
                    # ---- RELOCATE ----
                    new_a = ra[:pos] + ra[pos + 1:]
                    if rng.random() < 0.03:
                        if len(ra) == 1 or len(routes) >= max_veh:
                            continue
                        changes = [(a, new_a), (None, [cid])]
                    elif b == a:
                        q = rng.randrange(len(new_a) + 1)
                        if nb is not None and nb in new_a:
                            q = new_a.index(nb) + rng.randrange(2)
                        changes = [(a, new_a[:q] + [cid] + new_a[q:])]
                    else:
                        rb = routes[b]
                        q = rng.randrange(len(rb) + 1)
                        if nb is not None:
                            q = rb.index(nb) + rng.randrange(2)
                        changes = [(a, new_a), (b, rb[:q] + [cid] + rb[q:])]

                elif m < 0.50:
                    # ---- SWAP ----
                    if b == a:
                        if len(ra) < 2:
                            continue
                        if nb is not None and nb in ra:
                            j = ra.index(nb)
                        else:
                            j = rng.randrange(len(ra))
                        if j == pos:
                            continue
                        new_a = ra[:]
                        new_a[pos], new_a[j] = new_a[j], new_a[pos]
                        changes = [(a, new_a)]
                    else:
                        rb = routes[b]
                        j = rb.index(nb) if nb is not None else rng.randrange(len(rb))
                        new_a, new_b = ra[:], rb[:]
                        new_a[pos], new_b[j] = rb[j], ra[pos]
                        changes = [(a, new_a), (b, new_b)]

                elif m < 0.70 and use_oropt:
                    # ---- OR-OPT (segment de 2 ou 3 clients) ----
                    L = 2 + rng.randrange(2)
                    if len(ra) < L:
                        continue
                    s = min(pos, len(ra) - L)
                    seg = ra[s:s + L]
                    if rng.random() < 0.3:
                        seg = seg[::-1]
                    rest = ra[:s] + ra[s + L:]
                    if b == a:
                        q = rng.randrange(len(rest) + 1)
                        if nb is not None and nb in rest:
                            q = rest.index(nb) + rng.randrange(2)
                        changes = [(a, rest[:q] + seg + rest[q:])]
                    else:
                        rb = routes[b]
                        q = rng.randrange(len(rb) + 1)
                        if nb is not None:
                            q = rb.index(nb) + rng.randrange(2)
                        changes = [(a, rest), (b, rb[:q] + seg + rb[q:])]

                elif m < 0.95 and use_2opt_star:
                    # ---- 2-opt* : echange des queues ----
                    if b == a:
                        continue
                    rb = routes[b]
                    j = rb.index(nb) if nb is not None else rng.randrange(len(rb))
                    i = pos + 1  # on relie cid -> nb
                    new_a = ra[:i] + rb[j:]
                    new_b = rb[:j] + ra[i:]
                    changes = [(a, new_a), (b, new_b)]

                else:
                    # ---- 2-opt intra-tournee ----
                    if len(ra) < 3:
                        continue
                    i, j = sorted(rng.sample(range(len(ra)), 2))
                    new_a = ra[:i] + ra[i:j + 1][::-1] + ra[j + 1:]
                    changes = [(a, new_a)]

            if not changes:
                continue

            # ---------------- faisabilite + delta ----------------
            delta = 0.0
            new_dists = []
            d_routes = 0
            ok = True
            for idx, route in changes:
                nd = ev(route)
                if nd is None:
                    ok = False
                    break
                new_dists.append(nd)
                if idx is None:
                    d_routes += 1
                else:
                    delta += nd - dists[idx]
                    if not route:
                        d_routes -= 1
            if not ok:
                continue
            for idx, nd in zip((c[0] for c in changes), new_dists):
                if idx is None:
                    delta += nd
            delta += lam * d_routes

            # ---------------- Metropolis ----------------
            if delta < 0 or rng.random() < math.exp(-delta / temperature):
                empty = False
                for (idx, route), nd in zip(changes, new_dists):
                    if idx is None:
                        routes.append(route)
                        dists.append(nd)
                        idx = len(routes) - 1
                    else:
                        routes[idx] = route
                        dists[idx] = nd
                    if route:
                        for c in route:
                            route_of[c] = idx
                    else:
                        empty = True
                if empty:
                    keep = [k for k, rr in enumerate(routes) if rr]
                    routes = [routes[k] for k in keep]
                    dists = [dists[k] for k in keep]
                    rebuild_map()

                current += delta
                if current < best - 1e-9:
                    best = current
                    best_routes = [rr[:] for rr in routes]

            if it % 500 == 0:
                history.append((it, sum(ev(rr) for rr in best_routes)))

    execution_time = time.time() - start
    return {
        "routes": best_routes,
        "distance": sum(ev(r) for r in best_routes),
        "vehicles": len(best_routes),
        "execution_time": execution_time,
        "history": history,
    }


def _eliminate(routes, dists, ev, rng):
    """C. Tente de vider une petite tournee : chaque client va a sa meilleure
    position faisable dans une autre tournee. Retourne la liste 'changes' ou None."""
    if len(routes) < 2:
        return None
    order = sorted(range(len(routes)), key=lambda k: len(routes[k]))
    a = order[rng.randrange(min(2, len(order)))]
    victims = routes[a][:]
    rng.shuffle(victims)

    work = {}  # idx -> route modifiee
    for cid in victims:
        best_gain, best_k, best_route = None, None, None
        for k in range(len(routes)):
            if k == a:
                continue
            rk = work.get(k, routes[k])
            base = ev(rk)
            for q in range(len(rk) + 1):
                cand = rk[:q] + [cid] + rk[q:]
                nd = ev(cand)
                if nd is not None:
                    inc = nd - base
                    if best_gain is None or inc < best_gain:
                        best_gain, best_k, best_route = inc, k, cand
        if best_k is None:
            return None
        work[best_k] = best_route

    changes = [(a, [])]
    changes += [(k, r) for k, r in work.items()]
    return changes


import math
import time


def calculate_distance(c1, c2):
    return math.sqrt(
        (c1["x"] - c2["x"]) ** 2 +
        (c1["y"] - c2["y"]) ** 2
    )


def calculate_route_distance(route, customers, depot):
    if not route:
        return 0

    distance = calculate_distance(depot, customers[route[0]])

    for i in range(len(route) - 1):
        distance += calculate_distance(
            customers[route[i]],
            customers[route[i + 1]]
        )

    distance += calculate_distance(customers[route[-1]], depot)

    return distance


def check_route_feasibility(route, customers, depot, capacity):
    current_time = depot["ready_time"]
    current_load = 0
    current_location = depot

    for customer_id in route:
        customer = customers[customer_id]

        arrival_time = current_time + calculate_distance(
            current_location, customer
        )

        service_start = max(arrival_time, customer["ready_time"])

        if service_start > customer["due_time"]:
            return False

        current_load += customer["demand"]

        if current_load > capacity:
            return False

        current_time = service_start + customer["service_time"]
        current_location = customer

    return_time = current_time + calculate_distance(current_location, depot)

    return return_time <= depot["due_time"]


def solomon_insertion(instance):
    start_time = time.time()

    depot = instance["depot"]
    capacity = instance["capacity"]

    customers = {
        c["id"]: c for c in instance["customers"]
    }

    unassigned = set(customers.keys())
    routes = []

    while unassigned:
        seed = max(
            unassigned,
            key=lambda i: calculate_distance(depot, customers[i])
        )

        route = [seed]
        unassigned.remove(seed)

        while True:
            best_customer = None
            best_position = None
            best_cost = float("inf")

            for customer_id in unassigned:
                for position in range(len(route) + 1):
                    new_route = route.copy()
                    new_route.insert(position, customer_id)

                    if check_route_feasibility(
                        new_route, customers, depot, capacity
                    ):
                        old_cost = calculate_route_distance(
                            route, customers, depot
                        )
                        new_cost = calculate_route_distance(
                            new_route, customers, depot
                        )
                        insertion_cost = new_cost - old_cost

                        if insertion_cost < best_cost:
                            best_cost = insertion_cost
                            best_customer = customer_id
                            best_position = position

            if best_customer is None:
                break

            route.insert(best_position, best_customer)
            unassigned.remove(best_customer)

        routes.append(route)

    execution_time = time.time() - start_time

    total_distance = sum(
        calculate_route_distance(route, customers, depot)
        for route in routes
    )

    return {
        "routes": routes,
        "distance": total_distance,
        "vehicles": len(routes),
        "execution_time": execution_time
    }


def clarke_wright(instance):
    start_time = time.time()

    depot = instance["depot"]
    capacity = instance["capacity"]
    max_vehicles = instance["vehicles"]

    customers = {
        c["id"]: c for c in instance["customers"]
    }

    # Initial solution: one route per customer
    routes = [[customer_id] for customer_id in customers]

    # Calculate savings
    savings = []
    customer_ids = list(customers.keys())

    for i in range(len(customer_ids)):
        for j in range(i + 1, len(customer_ids)):
            ci = customer_ids[i]
            cj = customer_ids[j]

            saving = (
                calculate_distance(depot, customers[ci])
                + calculate_distance(depot, customers[cj])
                - calculate_distance(customers[ci], customers[cj])
            )

            savings.append((saving, ci, cj))

    savings.sort(reverse=True)

    # Merge routes according to savings
    for saving, ci, cj in savings:

        route_i = None
        route_j = None

        for route in routes:
            if ci in route:
                route_i = route
            if cj in route:
                route_j = route

        if route_i is None or route_j is None:
            continue

        if route_i is route_j:
            continue

        # Both customers must be at route endpoints
        if ci not in (route_i[0], route_i[-1]):
            continue

        if cj not in (route_j[0], route_j[-1]):
            continue

        # First possible orientation
        first_route = (
            route_i if route_i[-1] == ci
            else route_i[::-1]
        )

        second_route = (
            route_j if route_j[0] == cj
            else route_j[::-1]
        )

        candidate_1 = first_route + second_route

        # Reverse orientation
        candidate_2 = candidate_1[::-1]

        feasible_candidates = []

        for candidate in [candidate_1, candidate_2]:
            if check_route_feasibility(
                candidate, customers, depot, capacity
            ):
                distance = calculate_route_distance(
                    candidate, customers, depot
                )
                feasible_candidates.append((distance, candidate))

        if not feasible_candidates:
            continue

        # Select the feasible candidate
        best_route = min(
            feasible_candidates,
            key=lambda x: x[0]
        )[1]

        # Replace old routes with merged route
        index_i = routes.index(route_i)
        index_j = routes.index(route_j)

        for index in sorted([index_i, index_j], reverse=True):
            routes.pop(index)

        routes.append(best_route)

    execution_time = time.time() - start_time

    total_distance = sum(
        calculate_route_distance(route, customers, depot)
        for route in routes
    )

    return {
        "routes": routes,
        "distance": total_distance,
        "vehicles": len(routes),
        "execution_time": execution_time,
        "feasible": len(routes) <= max_vehicles
    }

from collections import Counter


def validate_solution(instance, result):
    customers = {
        c["id"]: c for c in instance["customers"]
    }

    depot = instance["depot"]
    capacity = instance["capacity"]
    routes = result["routes"]

    # Check that every customer is visited exactly once
    visited = [
        customer_id
        for route in routes
        for customer_id in route
    ]

    counts = Counter(visited)

    missing = set(customers.keys()) - set(visited)
    duplicates = [
        customer_id
        for customer_id, count in counts.items()
        if count > 1
    ]

    unknown = set(visited) - set(customers.keys())

    # Check capacity and time-window feasibility of each route
    invalid_routes = []

    for i, route in enumerate(routes, start=1):
        if not check_route_feasibility(
            route, customers, depot, capacity
        ):
            invalid_routes.append(i)

    # Check vehicle limit
    vehicle_limit_ok = len(routes) <= instance["vehicles"]

    return {
        "all_customers_served_once": (
            not missing and not duplicates and not unknown
        ),
        "missing_customers": sorted(missing),
        "duplicate_customers": duplicates,
        "unknown_customers": sorted(unknown),
        "all_routes_feasible": len(invalid_routes) == 0,
        "invalid_routes": invalid_routes,
        "vehicle_limit_respected": vehicle_limit_ok,
        "solution_feasible": (
            not missing
            and not duplicates
            and not unknown
            and len(invalid_routes) == 0
            and vehicle_limit_ok
        )
    }


import csv
import os

from data_loader import load_homberger_instance
from heuristic_method import clarke_wright, validate_solution


# Load the large instance (200 customers)
instance = load_homberger_instance("C1_2_1.TXT")

# Execute Clarke & Wright
result = clarke_wright(instance)

# Validate the solution
validation = validate_solution(instance, result)


# Display results
print("\n========== CLARKE & WRIGHT RESULTS ==========")

print("Instance:", instance["name"])
print("Number of customers:", len(instance["customers"]))
print("Number of vehicles used:", result["vehicles"])
print("Total distance:", round(result["distance"], 2))
print("Execution time:", round(result["execution_time"], 4), "seconds")

print("\n========== VALIDATION RESULTS ==========")

print(
    "All customers served once:",
    validation["all_customers_served_once"]
)
print("Missing customers:", validation["missing_customers"])
print("Duplicate customers:", validation["duplicate_customers"])
print("Unknown customers:", validation["unknown_customers"])
print("All routes feasible:", validation["all_routes_feasible"])
print("Invalid routes:", validation["invalid_routes"])
print("Vehicle limit respected:", validation["vehicle_limit_respected"])
print("Solution feasible:", validation["solution_feasible"])

print("\nRoutes:")
for i, route in enumerate(result["routes"], start=1):
    print(f"Vehicle {i}: {route}")

print("=============================================")


# Save results in CSV
os.makedirs("results", exist_ok=True)

file_path = "results/heuristic_results.csv"
file_exists = os.path.isfile(file_path)

with open(file_path, "a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    if not file_exists:
        writer.writerow([
            "Instance",
            "Method",
            "Customers",
            "Distance",
            "Vehicles",
            "Execution_Time",
            "Routes"
        ])

    writer.writerow([
        instance["name"],
        "Clarke & Wright",
        len(instance["customers"]),
        round(result["distance"], 2),
        result["vehicles"],
        round(result["execution_time"], 6),
        str(result["routes"])
    ])

print("\nResults saved in:", file_path)

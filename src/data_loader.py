
import json
from pathlib import Path


# Racine du projet
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "datasets"


def find_file(filename, folder_hint=None):
    """Recherche un fichier dans le dossier datasets."""

    for path in DATA_DIR.rglob("*"):
        if path.is_file() and path.name.lower() == filename.lower():

            if folder_hint is None or folder_hint.lower() in str(path).lower():
                return path

    raise FileNotFoundError(f"Fichier introuvable : {filename}")


def load_json_instance(filename, folder_hint):
    """Lecture d'une instance Solomon au format JSON."""

    path = find_file(filename, folder_hint)

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    vehicle = data["vehicle"]
    nodes = data["customers"]

    depot = next(node for node in nodes if node["id"] == 0)
    customers = [node for node in nodes if node["id"] != 0]

    return {
        "name": filename,
        "format": "JSON",
        "vehicles": vehicle["number"],
        "capacity": vehicle["capacity"],
        "depot": depot,
        "customers": customers
    }


def load_homberger_instance(filename):
    """Lecture d'une instance Homberger au format TXT."""

    path = find_file(filename)

    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    vehicle_number = None
    capacity = None
    nodes = []
    reading_customers = False

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if "CUST NO." in line:
            reading_customers = True
            continue

        if not reading_customers:
            continue

        parts = line.split()

        if len(parts) == 7:
            try:
                values = [int(value) for value in parts]
                nodes.append({
                    "id": values[0],
                    "x": values[1],
                    "y": values[2],
                    "demand": values[3],
                    "ready_time": values[4],
                    "due_time": values[5],
                    "service_time": values[6]
                })
            except ValueError:
                continue

    # Lecture des informations du véhicule
    for i, line in enumerate(lines):
        if "NUMBER" in line and "CAPACITY" in line:
            values = lines[i + 1].split()
            vehicle_number = int(values[0])
            capacity = int(values[1])
            break

    if not nodes:
        raise ValueError("Aucune donnée client trouvée.")

    depot = next(node for node in nodes if node["id"] == 0)
    customers = [node for node in nodes if node["id"] != 0]

    return {
        "name": filename,
        "format": "TXT",
        "vehicles": vehicle_number,
        "capacity": capacity,
        "depot": depot,
        "customers": customers
    }


def display_instance(instance):

    print("\n========== INSTANCE INFORMATION ==========")

    print("Name:", instance["name"])
    print("Format:", instance["format"])
    print("Number of vehicles:", instance["vehicles"])
    print("Vehicle capacity:", instance["capacity"])
    print("Number of customers:", len(instance["customers"]))

    print("\nDepot:")
    print(instance["depot"])

    print("\nFirst 3 customers:")
    for customer in instance["customers"][:3]:
        print(customer)

    print("==========================================\n")


if __name__ == "__main__":

    small = load_json_instance("C101.json", "solomon_25")
    medium = load_json_instance("C101.json", "solomon_50")
    large = load_homberger_instance("C1_2_1.TXT")

    display_instance(small)
    display_instance(medium)
    display_instance(large)

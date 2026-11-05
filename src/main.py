from data_loader import load_homberger_instance
from exact_method import solve_exact

instance = load_homberger_instance("C1_2_1.TXT")

result = solve_exact(
    instance,
    time_limit=300,
    instance_name="Homberger_200_C1_2_1"
)
# agents/tools.py
from langchain.tools import tool
import math

# TODO: Implement the Haversine formula to calculate the distance between two coordinates (lat/lon).
# Remember to consistently return the distance in kilometers or miles.
@tool
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates the distance in km between two geographic points."""
    pass

# TODO: Implement logic to check if travel speed is humanly possible.
# Divide distance (km) by time difference (hours). If > 900 km/h, return True (it's fraud/impossible).
@tool
def check_impossible_travel(dist_km: float, time_diff_hours: float) -> bool:
    """Verifies if the travel between two transactions is physically impossible."""
    pass

# TODO: Create a function that takes a user's history and returns how many transactions were made in the last X hours.
@tool
def get_transactions_last_n_hours(user_id: str, hours: int) -> int:
    """Returns the number of recent transactions by a user within a given hour range."""
    pass

# TODO: Create a function to calculate the time elapsed (in minutes) since the user's last transaction.
@tool
def time_since_last_transaction(user_id: str, current_timestamp: str) -> float:
    """Calculates the minutes since the last purchase."""
    pass

# TODO: Implement user's average spending calculation (mean).
# Consider caching this result to avoid iterating through the whole list on every call.
@tool
def get_user_average_spending(user_id: str) -> float:
    """Calculates the historical average spending of the user."""
    pass

# TODO: Function that compares the current amount vs historical average and returns a multiplier (e.g., 10x larger).
@tool
def calculate_amount_anomaly(user_id: str, current_amount: float) -> float:
    """Returns how unusually large the current amount is compared to the historical average."""
    pass

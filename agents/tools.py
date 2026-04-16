# agents/tools.py
from langchain.tools import tool
import pandas as pd
import math

# =====================================================================
# IN-MEMORY DATASTORE
# This dictionary is populated during the test/preprocessing phase.
# Structure: {'user_id': DataFrame(columns: amount, timestamp, lat, lng, etc.)}
# =====================================================================
USER_DATA = {} 

@tool
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates the great-circle distance in km between two geographic points using Haversine."""
    R = 6371.0 
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return float(round(R * c, 2))

@tool
def check_impossible_travel(dist_km: float, time_diff_hours: float) -> bool:
    """
    Verifies if travel is physically impossible. 
    Returns True if the required speed exceeds 900 km/h (average jet speed).
    """
    if time_diff_hours <= 0:
        # If distance is significant but time is zero, it's impossible (teleportation)
        return True if dist_km > 1.0 else False
    
    speed = dist_km / time_diff_hours
    return speed > 900.0

@tool
def get_last_known_location(user_id: str, current_timestamp: str) -> dict:
    """
    Retrieves the location (lat/lng) and time of the user's transaction 
    immediately preceding the current one. Vital for distance checks.
    """
    df = USER_DATA.get(user_id)
    if df is None or df.empty:
        return {}

    # Ensure timestamp is datetime for comparison
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    curr_ts = pd.to_datetime(current_timestamp, utc=True)
    
    # Filter for past transactions only and get the latest one
    past_df = df[df['timestamp'] < curr_ts].sort_values('timestamp', ascending=False)
    
    if past_df.empty:
        return {}
        
    last_tx = past_df.iloc[0]
    return {
        "prev_lat": float(last_tx['lat']),
        "prev_lng": float(last_tx['lng']),
        "prev_timestamp": last_tx['timestamp'].isoformat()
    }

@tool
def calculate_amount_anomaly(user_id: str, current_amount: float, current_timestamp: str) -> float:
    """
    Calculates the Z-score of the current amount compared to historical mean.
    A Z-score > 3.0 is statistically anomalous.
    """
    df = USER_DATA.get(user_id)
    if df is None or df.empty:
        return 0.0
        
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    current_time = pd.to_datetime(current_timestamp, utc=True)
    past_df = df[df['timestamp'] < current_time]
    
    if len(past_df) < 2:
        return 0.0 

    mean_amount = past_df['amount'].mean()
    std_dev = past_df['amount'].std()

    if std_dev == 0:
        return 0.0 if current_amount == mean_amount else 99.0 

    z_score = (current_amount - mean_amount) / std_dev
    return float(round(z_score, 2))

@tool
def get_transactions_last_n_hours(user_id: str, hours: int, current_timestamp: str) -> int:
    """Counts how many transactions the user made in the N hours preceding the current one."""
    df = USER_DATA.get(user_id)
    if df is None or df.empty:
        return 0
        
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    current_time = pd.to_datetime(current_timestamp, utc=True)
    cutoff_time = current_time - pd.Timedelta(hours=hours)
    
    recent_tx = df[(df['timestamp'] >= cutoff_time) & (df['timestamp'] < current_time)]
    return int(len(recent_tx))

@tool
def time_since_last_transaction(user_id: str, current_timestamp: str) -> float:
    """Calculates minutes elapsed since the previous transaction."""
    df = USER_DATA.get(user_id)
    if df is None or df.empty:
        return -1.0
        
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    current_time = pd.to_datetime(current_timestamp, utc=True)
    past_df = df[df['timestamp'] < current_time]
    
    if past_df.empty:
        return -1.0
        
    last_tx_time = past_df['timestamp'].max()
    time_diff = current_time - last_tx_time
    
    return float(round(time_diff.total_seconds() / 60.0, 2))

@tool
def check_structuring_pattern(user_id: str, current_timestamp: str, current_amount: float, hours: int = 24) -> dict:
    """
    Analyzes 'Structuring/Smurfing' patterns. 
    Flags risk if aggregate volume in 24h exceeds 3x the user's average transaction.
    """
    df = USER_DATA.get(user_id)
    if df is None or df.empty:
        return {"structuring_risk": "Low"}

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    curr_ts = pd.to_datetime(current_timestamp, utc=True)
    
    window_start = curr_ts - pd.Timedelta(hours=hours)
    recent = df[(df['timestamp'] >= window_start) & (df['timestamp'] < curr_ts)]
    
    window_volume = recent['amount'].sum() + current_amount
    # Historical average excluding the current window for cleaner comparison
    hist_avg = df[df['timestamp'] < window_start]['amount'].mean()
    if pd.isna(hist_avg) or hist_avg == 0:
        hist_avg = df['amount'].mean() # Fallback

    ratio = window_volume / hist_avg if hist_avg > 0 else 1.0
    
    # Quant Logic: 3+ transactions totaling > 3x average is a clear structuring pattern
    is_suspicious = (len(recent) + 1 >= 3) and (ratio > 3.0)
    
    return {
        "tx_count_24h": int(len(recent) + 1),
        "total_volume_24h": float(round(window_volume, 2)),
        "volume_vs_avg_ratio": float(round(ratio, 2)),
        "structuring_risk": "High" if is_suspicious else "Low"
    }

# agents/tools.py (Añadir al final)

# Simulación de base de datos de usuarios (Cargada desde Users.csv en el preprocesador)
USER_PROFILES = {} 

@tool
def check_home_distance(user_id: str, current_lat: float, current_lng: float) -> dict:
    """
    Compares current transaction location with user's registered home address.
    """
    profile = USER_PROFILES.get(user_id)
    if not profile:
        return {"at_home": "unknown", "distance_km": -1}
    
    # Usamos la herramienta de distancia que ya tienes
    home_lat = profile.get('home_lat')
    home_lng = profile.get('home_lng')
    
    dist = calculate_distance(current_lat, current_lng, home_lat, home_lng)
    
    return {
        "distance_from_home_km": dist,
        "is_unusually_far": dist > 100.0 # Más de 100km de casa
    }

@tool
def check_iban_history(user_id: str, recipient_iban: str) -> dict:
    """
    Verifica si el IBAN destino ya ha sido usado por el usuario antes.
    """
    df = USER_DATA.get(user_id)
    if df is None or df.empty or 'recipient_iban' not in df.columns:
        return {"new_iban": True, "times_used_before": 0}
    
    count = len(df[df['recipient_iban'] == recipient_iban])
    return {
        "new_iban": count == 0,
        "times_used_before": count,
        "risk": "High" if count == 0 else "Low"
    }
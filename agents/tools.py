# agents/tools.py
from langchain.tools import tool
import pandas as pd
import math

# =====================================================================
# SIMULACIÓN DE DATASTORE EN MEMORIA
# El Preprocesador (Fase 1) debe poblar este diccionario al inicio.
# Estructura: {'user_id': DataFrame(columnas: amount, timestamp, etc.)}
# =====================================================================
USER_DATA = {} 

@tool
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates the great-circle distance in km between two geographic points."""
    R = 6371.0 
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

@tool
def check_impossible_travel(dist_km: float, time_diff_hours: float) -> bool:
    """Verifies if the travel between two transactions is physically impossible (> 900 km/h)."""
    if time_diff_hours <= 0:
        return True if dist_km > 1.0 else False
    return (dist_km / time_diff_hours) > 900.0

@tool
def calculate_amount_anomaly(user_id: str, current_amount: float, current_timestamp: str) -> float:
    """
    Calculates the Z-score of the current transaction amount.
    Returns how many standard deviations the amount is from the user's historical mean.
    """
    df = USER_DATA.get(user_id)
    if df is None or df.empty:
        return 0.0
        
    current_time = pd.to_datetime(current_timestamp)
    # Filtrar solo el historial estricto ANTES de esta transacción
    past_df = df[df['timestamp'] < current_time]
    
    if len(past_df) < 2:
        return 0.0 

    mean_amount = past_df['amount'].mean()
    std_dev = past_df['amount'].std()

    if std_dev == 0:
        return 0.0 if current_amount == mean_amount else 99.0 

    return (current_amount - mean_amount) / std_dev

@tool
def get_transactions_last_n_hours(user_id: str, hours: int, current_timestamp: str) -> int:
    """Returns the number of recent transactions by a user within a given hour range before the current transaction."""
    df = USER_DATA.get(user_id)
    if df is None or df.empty:
        return 0
        
    current_time = pd.to_datetime(current_timestamp)
    cutoff_time = current_time - pd.Timedelta(hours=hours)
    
    # Contar transacciones en la ventana de tiempo estricta
    recent_tx = df[(df['timestamp'] >= cutoff_time) & (df['timestamp'] < current_time)]
    return len(recent_tx)

@tool
def time_since_last_transaction(user_id: str, current_timestamp: str) -> float:
    """Calculates the minutes since the user's last transaction."""
    df = USER_DATA.get(user_id)
    if df is None or df.empty:
        return -1.0 # Indica que no hay historial previo
        
    current_time = pd.to_datetime(current_timestamp)
    past_df = df[df['timestamp'] < current_time]
    
    if past_df.empty:
        return -1.0
        
    last_tx_time = past_df['timestamp'].max() # Extrae la fecha más reciente del historial
    time_diff = current_time - last_tx_time
    
    return time_diff.total_seconds() / 60.0 # Retornamos en minutos

@tool
def check_structuring_pattern(user_id: str, current_timestamp: str, current_amount: float, hours: int = 24) -> dict:
    """
    Analyzes 'Smurfing/Structuring' by checking if the aggregate volume in a 
    sliding window deviates from the user's historical behavior.
    """
    df = USER_DATA.get(user_id)
    if df is None or df.empty:
        return {"structuring_risk": "Low", "reason": "No historical data"}

    # 1. Ensure datetime alignment
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    curr_ts = pd.to_datetime(current_timestamp, utc=True)
    
    # 2. Define the Window (Memory)
    # We look at the window PRIOR to the current transaction
    window_start = curr_ts - pd.Timedelta(hours=hours)
    recent_window = df[(df['timestamp'] >= window_start) & (df['timestamp'] < curr_ts)]
    
    # 3. Calculate Metrics
    window_count = len(recent_window) + 1 # +1 for the current transaction
    window_volume = recent_window['amount'].sum() + current_amount
    
    # 4. Statistical Logic (Quant Approach)
    # Instead of $500, we use a multiplier of the historical mean
    historical_avg_tx = df['amount'].mean()
    volume_to_avg_ratio = window_volume / historical_avg_tx if historical_avg_tx > 0 else 0
    
    # High Risk: 
    # - More than 3 transactions in the window 
    # - AND the total volume is > 10x their usual single transaction size
    is_suspicious = (window_count >= 3) and (volume_to_avg_ratio > 10)
    
    return {
        "analysis_window": f"{hours}h",
        "tx_count_in_window": int(window_count),
        "aggregate_volume": float(round(window_volume, 2)),
        "volume_vs_avg_ratio": float(round(volume_to_avg_ratio, 2)),
        "structuring_risk": "High" if is_suspicious else "Low",
        "logic": "Aggregated volume exceeds 10x historical average transaction size with high frequency."
    }
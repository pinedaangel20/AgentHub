"""utils/preprocessor.py
High-throughput fraud triage pre-processor.

This module provides a single public function ``preprocess_transactions`` that:
- Computes per-user behavioral baselines (average amount, std deviation, tx count) WITHOUT data leakage.
- Calculates rolling velocity windows (10 min and 24 h) per user.
- Derives ``time_since_last_tx`` and Z-score for the current amount.
- Flags new devices per user.
- Applies the triage logic to assign a ``category`` and builds a ``risk_context`` string.
- Returns only the rows that require LLM review (category == "llm_review").
"""

import pandas as pd
import numpy as np
from typing import List

# ── Feature Store (FIXED: Expanding window to prevent Lookahead Bias) ─────────

def _add_expanding_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Compute expanding baselines up to the CURRENT row to prevent data leakage."""
    df = df.sort_values(["user_id", "timestamp"]).copy()
    
    # Agrupamos por usuario
    grouped = df.groupby("user_id")["amount"]
    
    # Shift(1) asegura que la transacción actual NO se incluya en su propio promedio histórico
    shifted_amt = grouped.shift(1)
    
    # Calculamos métricas expansivas (históricas hasta ese momento)
    df["tx_count"] = df.groupby("user_id").cumcount()
    df["avg_amt"] = shifted_amt.groupby(df["user_id"]).expanding().mean().reset_index(level=0, drop=True).fillna(0.0)
    df["std_amt"] = shifted_amt.groupby(df["user_id"]).expanding().std().reset_index(level=0, drop=True).fillna(0.0)
    
    return df

# ── Velocity (FIXED: Numpy array to prevent MultiIndex alignment crash) ───────

def _add_rolling_velocities(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling transaction counts for 10-minute and 24-hour windows."""
    df = df.sort_values(["user_id", "timestamp"]).copy()
    temp_df = df.set_index("timestamp")

    def _rolling_count(df_local: pd.DataFrame, window: str) -> np.ndarray:
        return (
            df_local.groupby("user_id")["amount"]
            .rolling(window, closed="left")
            .count()
            .values # <- Extraemos values puros para evitar errores si hay timestamps duplicados
        )

    df["velocity_10min"] = np.nan_to_num(_rolling_count(temp_df, "10min"))
    df["velocity_24h"] = np.nan_to_num(_rolling_count(temp_df, "24h"))

    return df

# ── Time-since-last ──────────────────────────────────────────────────────────

def _add_time_since_last(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate seconds since the previous transaction for each user."""
    df = df.sort_values(["user_id", "timestamp"]).copy()
    df["prev_timestamp"] = df.groupby("user_id")["timestamp"].shift(1)
    df["time_since_last_tx"] = (
        (df["timestamp"] - df["prev_timestamp"]).dt.total_seconds()
    )
    df["time_since_last_tx"] = df["time_since_last_tx"].fillna(np.inf)
    df.drop(columns=["prev_timestamp"], inplace=True)
    return df

# ── Device flags ──────────────────────────────────────────────────────────────

def _add_device_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag a device as *new* for the user (first occurrence)."""
    df = df.sort_values(["user_id", "timestamp"]).copy()
    df["device_occurrence"] = df.groupby(["user_id", "device_id"]).cumcount()
    df["is_new_device"] = df["device_occurrence"] == 0
    df.drop(columns=["device_occurrence"], inplace=True)
    return df

# ── Z-score ───────────────────────────────────────────────────────────────────

def _add_z_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Z-score directly since baselines are already in the dataframe."""
    df["z_score"] = np.where(
        df["std_amt"] > 0,
        (df["amount"] - df["avg_amt"]) / df["std_amt"],
        0.0,
    )
    return df

# ── Triage logic ──────────────────────────────────────────────────────────────

def _apply_triage(df: pd.DataFrame) -> pd.DataFrame:
    """Assign ``category`` per the specification."""
    cond_approve = (
        (df["amount"] < 50)
        & (df["z_score"].abs() < 1.0)
        & (df["tx_count"] > 10)
    )
    cond_block = (
        (df["z_score"].abs() > 7.0)
        | ((df["velocity_10min"] > 5) & (df["tx_count"] < 3))
    )
    df["category"] = np.select(
        [cond_approve, cond_block],
        ["auto_approve", "auto_block"],
        default="llm_review",
    )
    return df

# ── Risk-context string ──────────────────────────────────────────────────────

def _build_risk_context(row: pd.Series) -> str:
    """Create a concise human-readable summary for the LLM."""
    parts: List[str] = []
    parts.append(
        f"User {row['user_id']} | avg=${row['avg_amt']:.2f}, "
        f"std=${row['std_amt']:.2f}, count={int(row['tx_count'])}"
    )
    parts.append(f"Amt=${row['amount']:.2f}, Z={row['z_score']:.2f}")
    parts.append(
        f"Vel10m={int(row['velocity_10min'] or 0)}, "
        f"Vel24h={int(row['velocity_24h'] or 0)}"
    )
    if np.isfinite(row["time_since_last_tx"]):
        parts.append(f"dt={int(row['time_since_last_tx'])}s")
    else:
        parts.append("dt=first_tx")
    if row.get("is_new_device"):
        parts.append("NEW_DEVICE")
    parts.append(f"Cat={row['category']}")
    return "; ".join(parts)

# ── Public API ────────────────────────────────────────────────────────────────

def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Main entry point."""
    required_cols = {"user_id", "amount", "timestamp", "device_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # El nuevo flujo ordenado y libre de bugs:
    df = _add_expanding_baselines(df)
    df = _add_rolling_velocities(df)
    df = _add_time_since_last(df)
    df = _add_device_flags(df)
    df = _add_z_score(df)
    df = _apply_triage(df)
    df["risk_context"] = df.apply(_build_risk_context, axis=1)

    result = df[df["category"] == "llm_review"].reset_index(drop=True)
    return result

# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = {
        "user_id": [1, 1, 1, 2, 2],
        "amount": [20.0, 55.0, 600.0, 30.0, 5.0],
        "timestamp": pd.date_range("2023-01-01", periods=5, freq="5min"),
        "device_id": ["d1", "d1", "d2", "d3", "d3"],
    }
    df_input = pd.DataFrame(data)
    llm_df = preprocess_transactions(df_input)
    print("=== Transactions flagged for LLM review ===")
    for _, row in llm_df.iterrows():
        print(f"  user={row['user_id']}  amt=${row['amount']:.2f}  ctx={row['risk_context']}")
    print(f"\nTotal: {len(llm_df)} of {len(df_input)} sent to LLM.")
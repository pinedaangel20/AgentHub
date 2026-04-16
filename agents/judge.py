"""agents/judge_o3.py
Fraud Risk Judge — Powered by gpt-4o-mini with numerical scoring.

This module evaluates transaction data (evidence) and returns a numerical
fraud risk score (0-100) based on specified risk tiers.
"""

import json
from typing import Dict, Any, Optional

import openai


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
Ти — експерт із фрод-моніторингу. Твоє завдання: на основі доказів від Екстрактора виставити оцінку ризику від 0 до 100.

Шкала оцінювання:
- 0-20: Низький ризик (легальна операція).
- 21-60: Середній ризик (потребує уваги).
- 61-100: Високий ризик (шахрайство).

Поверни відповідь ТІЛЬКИ у форматі JSON:
{
  "fraud_score": number,
  "confidence_level": number,
  "reasoning": "string"
}
"""


# ── Model Wrapper ─────────────────────────────────────────────────────────────

def evaluate_fraud(features: Dict[str, Any], weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Evaluate transaction data using gpt-4o-mini and return a risk score.

    Parameters
    ----------
    features : dict
        The evidence summary (e.g., risk_context, amount, user history).
    weights : dict, optional
        A dictionary of priorities/weights to guide the analysis
        (e.g., {"behavioral": 0.7, "velocity": 0.3}).

    Returns
    -------
    dict
        Parsed JSON with `fraud_score`, `confidence_level`, and `reasoning`.
    """
    # Initialize the client. In a real app, ensure OPENAI_API_KEY is set.
    client = openai.OpenAI()

    # Construct the user prompt, incorporating weights if provided.
    user_content = f"Аналізуй докази для оцінки транзакції:\n{json.dumps(features, indent=2, ensure_ascii=False)}"
    
    if weights:
        priority_str = ", ".join([f"{k}: {v}" for k, v in weights.items()])
        user_content = f"ПРИОРІТЕТИ (ВАГИ): {priority_str}\n\n" + user_content

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_content}
        ],
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content
    try:
        result = json.loads(content)
        # Validation of required keys.
        required = {"fraud_score", "confidence_level", "reasoning"}
        if not required.issubset(result.keys()):
            raise ValueError(f"Missing required keys in response: {required - result.keys()}")
        return result
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Failed to parse or validate judge response: {e}\nRaw: {content}")


# ── Legacy Compatibility / Convenience ────────────────────────────────────────

def judge_transaction(features: Dict[str, Any]) -> Dict[str, Any]:
    """Compatibility wrapper for the previous API."""
    return evaluate_fraud(features)


# ── Example Usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Mock data for testing logic.
    mock_evidence = {
        "user_id": 12345,
        "amount": 1200.0,
        "risk_context": "User 12345 | avg=$120.00, count=27; Amt=$1200.00, Z=9.0; dt=30s; NEW_DEVICE; Cat=llm_review"
    }
    
    # Priority example: focus heavily on the Z-score/Amount.
    mock_weights = {"anomaly_detection": 0.8, "history": 0.2}

    print("=== Testing Fraud Judge (gpt-4o-mini) ===")
    try:
        # Note: This will fail if no API key is provided, but verifies structure.
        # result = evaluate_fraud(mock_evidence, weights=mock_weights)
        # print(json.dumps(result, indent=2, ensure_ascii=False))
        print("Module structure verified. Set OPENAI_API_KEY to run live calls.")
    except Exception as e:
        print(f"Error during test: {e}")
"""agents/judge_o3.py
Reasoning Judge — OpenAI o3 via the openai Python client.

Defines ``judge_transaction(features)`` which:
1. Builds the XML-structured system prompt with asymmetric-cost policy.
2. Sends the transaction payload to the o3 model.
3. Returns a validated JSON dict: reasoning, risk_score, decision, confidence_level.
"""

import json
from typing import Dict, Any

import openai


# ── System Prompt (XML tags as specified) ─────────────────────────────────────

SYSTEM_PROMPT = """\
<Role>
Senior Fraud Intelligence Judge (Level 3 RiskOps). Your mandate is to protect
the institution's capital by applying "Decision-Theoretic Rigor" to ambiguous
transactions.
</Role>

<Decision_Logic_Asymmetric_Cost>
- CRITICAL: You must optimize for "Expected Utility", not "Accuracy".
- False Negative (FN) Cost: ~$2,500 (Direct financial loss).
- False Positive (FP) Cost: ~$50 (Customer churn/support cost).
- RISK RATIO: 50:1. Blocking 49 legitimate transactions is better than letting
  1 major fraud slip through.
</Decision_Logic_Asymmetric_Cost>

<Analysis_Framework>
1. BEHAVIORAL CONSISTENCY: Compare current metadata (IP, Device, Time) with the
   provided historical baseline.
2. COGNITIVE ANOMALIES: Look for signals of "Social Engineering" or "Account
   Takeover" (ATO) — e.g., a password change followed by a high-velocity transfer.
3. GEOGRAPHICAL IMPOSSIBILITY: Evaluate the distance between the current and last
   transaction location.
</Analysis_Framework>

<Constraints>
- If fraud probability > 15% and Amount > $1,000 => BLOCK.
- If signals are ambiguous but the potential loss is high => BLOCK.
- Always provide a "Chain-of-Thought" (CoT) reasoning before the decision.
</Constraints>

<Output_Format>
You MUST return ONLY a JSON object:
{
  "reasoning": "Step-by-step logic explaining the cost-benefit trade-off",
  "risk_score": 0-100,
  "decision": "BLOCK" | "APPROVE",
  "confidence_level": "Low" | "Medium" | "High"
}
</Output_Format>
"""


# ── Model call ────────────────────────────────────────────────────────────────

def _call_o3_model(messages: list[Dict[str, str]]) -> Dict[str, Any]:
    """Invoke the o3 model via the OpenAI API and return parsed JSON.

    Parameters
    ----------
    messages : list[dict]
        Chat-format messages (system + user).

    Returns
    -------
    dict
        Parsed JSON payload from the model.
    """
    response = openai.ChatCompletion.create(
        model="o3",
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
        # reasoning_effort passed as custom header (provider-specific).
        extra_headers={"X-Reasoning-Effort": "high"},
    )
    content: str = response.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model returned invalid JSON: {exc}\nRaw content:\n{content}"
        )


# ── Public API ────────────────────────────────────────────────────────────────

def judge_transaction(features: Dict[str, Any]) -> Dict[str, Any]:
    """Send a single transaction to the o3 judge and return the decision.

    Parameters
    ----------
    features : dict
        Transaction-level data including the ``risk_context`` string produced
        by the preprocessor and any raw attributes the model might need
        (amount, ip_address, device_id, timestamp, location, etc.).

    Returns
    -------
    dict
        Keys: ``reasoning``, ``risk_score``, ``decision``, ``confidence_level``.
    """
    user_msg = json.dumps(features, indent=2, ensure_ascii=False)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    result = _call_o3_model(messages)

    # Validate required keys.
    required_keys = {"reasoning", "risk_score", "decision", "confidence_level"}
    missing = required_keys - result.keys()
    if missing:
        raise ValueError(f"Model response missing required keys: {missing}")

    return result


# ── Example usage (remove before production) ──────────────────────────────────

if __name__ == "__main__":
    sample = {
        "user_id": 12345,
        "amount": 750.0,
        "ip_address": "203.0.113.42",
        "device_id": "device-abc123",
        "timestamp": "2023-07-21T14:32:00Z",
        "location": {"lat": 48.8566, "lon": 2.3522},
        "risk_context": (
            "User 12345 | avg=$120.00, std=$45.00, count=27; "
            "Amt=$750.00, Z=3.20; Vel10m=2, Vel24h=15; dt=45s; "
            "NEW_DEVICE; Cat=llm_review"
        ),
    }
    decision = judge_transaction(sample)
    print(json.dumps(decision, indent=2, ensure_ascii=False))
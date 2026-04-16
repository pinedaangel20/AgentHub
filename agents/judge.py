"""agents/judge_o3.py
Reasoning Judge — OpenAI o3 via the openai Python client.

Defines ``judge_transaction(features)`` which:
1. Builds the XML-structured system prompt with asymmetric-cost policy.
2. Sends the transaction payload to the o3 model.
3. Returns a validated JSON dict: reasoning, risk_score, decision, confidence_level.
"""

import json
import os
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langfuse import observe
from langfuse.langchain import CallbackHandler


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
judge_model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/o3-mini", # NOTA: OpenRouter usa 'openai/o3-mini' para acceder a o3
    temperature=0.2,
    max_tokens=1024,
    # reasoning_effort is supported in OpenRouter via extra_body
    model_kwargs={"extra_body": {"reasoning_effort": "high"}}
)

# ── Public API ────────────────────────────────────────────────────────────────

@observe(name="Agent_Judge_o3", as_type="generation")
def judge_transaction(features: Dict[str, Any]) -> Dict[str, Any]:
    """Send a single transaction to the o3 judge and return the decision."""
    
    user_msg = json.dumps(features, indent=2, ensure_ascii=False)
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg)
    ]

    langfuse_handler = CallbackHandler()

    try:
        response = judge_model.invoke(
            messages,
            config={"callbacks": [langfuse_handler]}
        )
        content = response.content.strip()
        
        # Parse the JSON response
        clean_content = content.replace('```json', '').replace('```', '')
        result = json.loads(clean_content)
        
        # Validate required keys
        required_keys = {"reasoning", "risk_score", "decision", "confidence_level"}
        missing = required_keys - result.keys()
        if missing:
             raise ValueError(f"Model response missing required keys: {missing}")
             
        return result

    except Exception as exc:
        raise ValueError(f"Failed to execute Judge model or parse JSON: {exc}")


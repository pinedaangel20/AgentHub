# agents/judge.py
import json
import os
from typing import Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langfuse import observe
from langfuse.langchain import CallbackHandler

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
Ти — експерт із фрод-моніторингу. Твоє завдання: на основі доказів від Екстрактора виставити оцінку ризику від 0 до 100 та прийняти рішення (BLOCK або APPROVE).

Шкала оцінювання:
- 0-20: Низький ризик (легальна операція). -> APPROVE
- 21-60: Середній ризик (потребує уваги). -> APPROVE (або BLOCK якщо сума дуже велика)
- 61-100: Високий ризик (шахрайство). -> BLOCK

CRITICAL RISK MULTIPLIERS (Apply these rules strictly):
1. IMPOSSIBLE TRAVEL: If evidence shows 'is_physically_possible: false', this is a smoking gun. Immediately score 85+ and return BLOCK.
2. DEEP SLEEP WINDOW: Look at the timestamp of the transaction. If the local time is between 02:00 AM and 05:00 AM, add +30 to the fraud score (High probability of victim sleeping).
3. MERCHANT RISK: If the merchant category is High Risk (Crypto, Casino, Jewelry), multiply the risk significance.

Поверни відповідь ТІЛЬКИ у форматі JSON:
{
  "fraud_score": number,
  "decision": "BLOCK" | "APPROVE",
  "confidence_level": number,
  "reasoning": "string"
}
"""

# ── Model Wrapper ─────────────────────────────────────────────────────────────

# Inicializamos el modelo correctamente con OpenRouter y LangChain
judge_model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="gpt-4o-mini",
    temperature=0.0
)

@observe(name="Agent_Judge", as_type="generation")
def evaluate_fraud(features: Dict[str, Any], weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Evaluate transaction data using LangChain + OpenRouter."""
    
    user_content = f"Аналізуй докази для оцінки транзакції:\n{json.dumps(features, indent=2, ensure_ascii=False)}"
    
    if weights:
        priority_str = ", ".join([f"{k}: {v}" for k, v in weights.items()])
        user_content = f"ПРИОРІТЕТИ (ВАГИ): {priority_str}\n\n" + user_content

    messages = [
        SystemMessage(content=SYSTEM_PROMPT.strip()),
        HumanMessage(content=user_content)
    ]

    langfuse_handler = CallbackHandler()

    try:
        response = judge_model.invoke(
            messages,
            config={"callbacks": [langfuse_handler]}
        )
        
        content = response.content.strip()
        clean_content = content.replace('```json', '').replace('```', '')
        result = json.loads(clean_content)
        
        # Validamos que devuelva la decisión para que test.py no explote
        required = {"fraud_score", "decision", "confidence_level", "reasoning"}
        if not required.issubset(result.keys()):
            raise ValueError(f"Missing required keys in response: {required - result.keys()}")
            
        return result
        
    except Exception as e:
        raise RuntimeError(f"Failed to execute Judge model or parse JSON: {e}")

# ── Legacy Compatibility / Convenience ────────────────────────────────────────

def judge_transaction(features: Dict[str, Any]) -> Dict[str, Any]:
    """Compatibility wrapper for the main API."""
    return evaluate_fraud(features)
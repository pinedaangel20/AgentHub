# agents/judge.py
import json
import os
from typing import Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langfuse import observe
from langfuse.langchain import CallbackHandler

# ── System Prompt (ACTUALIZADO CON REGLAS DE FUSIÓN) ──────────────────────────

SYSTEM_PROMPT = """
Ти — експерт із фрод-моніторингу. Твоє завдання: на основі доказів (Math & Text Evidence) виставити оцінку ризику від 0 до 100 та прийняти рішення (BLOCK або APPROVE).

Шкала оцінювання:
- 0-20: Низький ризик -> APPROVE
- 21-70: Середній ризик -> APPROVE (BLOCK якщо сума > 1000)
- 71-100: Високий ризик -> BLOCK

### CRITICAL RISK MULTIPLIERS (Strict Rules):

1. SOCIAL ENGINEERING (NEW): Якщо 'text_evidence' показує 'social_engineering_detected: true', це автоматично 90+ балів. Phishing = BLOCK.
2. IMPOSSIBLE TRAVEL: Якщо 'is_physically_possible: false', негайно став 90+ балів та BLOCK.
3. HOME ADDRESS PROXIMITY (NEW): Якщо транзакція 'in-person' і знаходиться далі ніж 200км від дому ('distance_from_home_km'), додай +40 до ризику.
4. NEW IBAN (NEW): Для банківських переказів, якщо IBAN є новим для користувача, додай +30 до ризику.
5. DEEP SLEEP WINDOW: Якщо транзакція між 02:00 та 05:00 за місцевим часом, додай +20.
6. MERCHANT RISK: Категорії Crypto, Casino, Jewelry мають найвищий пріоритет ризику.

Поверни відповідь ТІЛЬКИ у форматі JSON:
{
  "fraud_score": number,
  "decision": "BLOCK" | "APPROVE",
  "confidence_level": number,
  "reasoning": "string"
}
"""

# ── Model Wrapper ─────────────────────────────────────────────────────────────

judge_model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None,
    model="gpt-4o-mini", # Sugerencia: Para el juez final, o3-mini es más potente si el presupuesto lo permite
    temperature=0.0
)

@observe(name="Agent_Judge", as_type="generation")
def evaluate_fraud(features: Dict[str, Any], weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Evaluate transaction data using LangChain + OpenRouter."""
    
    # Fusionamos toda la evidencia en el prompt
    user_content = f"Аналізуй докази (Math Evidence + Text Evidence) для оцінки транзакції:\n{json.dumps(features, indent=2, ensure_ascii=False)}"
    
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
        # Limpieza robusta de JSON
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        
        result = json.loads(content)
        
        required = {"fraud_score", "decision", "confidence_level", "reasoning"}
        if not required.issubset(result.keys()):
            raise ValueError(f"Missing required keys: {required - result.keys()}")
            
        return result
        
    except Exception as e:
        # Fallback de seguridad en caso de error de parsing
        return {
            "fraud_score": 100, 
            "decision": "BLOCK", 
            "confidence_level": 0, 
            "reasoning": f"System error during judging: {str(e)}"
        }

# ── Legacy Compatibility ──────────────────────────────────────────────────────

def judge_transaction(features: Dict[str, Any]) -> Dict[str, Any]:
    """Compatibility wrapper for the main API."""
    return evaluate_fraud(features)
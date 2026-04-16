# agents/extractor_4omini.py
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langfuse.decorators import observe  # <-- Importado correctamente arriba

# Importamos TUS herramientas
from agents.tools import (
    calculate_distance, 
    check_impossible_travel, 
    calculate_amount_anomaly, 
    get_transactions_last_n_hours, 
    time_since_last_transaction
)

# 1. Definimos las herramientas disponibles para este agente
tools = [
    calculate_distance, 
    check_impossible_travel, 
    calculate_amount_anomaly, 
    get_transactions_last_n_hours, 
    time_since_last_transaction
]

# 2. Inicializamos el modelo (rápido y barato para usar herramientas)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# Vinculamos las herramientas al modelo de forma nativa
llm_with_tools = llm.bind_tools(tools)

# 3. El System Prompt (CRÍTICO para FinMath)
SYSTEM_PROMPT = """
You are the strictly mathematical 'Evidence Extractor' agent for a fraud detection system.
Your ONLY job is to execute the specific analytical tools requested by the Orchestrator, 
read the mathematical results, and output a raw JSON summary.

RULES:
1. DO NOT guess or calculate numbers yourself. ALWAYS invoke the provided tools.
2. If calculating spatial anomalies, you MUST use `calculate_distance` first, then `check_impossible_travel`.
3. If evaluating amounts, use `calculate_amount_anomaly`. If Z-score > 3.0, flag as highly anomalous.
4. Your final output MUST be ONLY a valid JSON object. No markdown formatting (```json), no conversational text.

Expected Output Format Example:
{
  "agent_id": "math_extractor_4o_mini",
  "impossible_travel_detected": true,
  "distance_km": 1500.5,
  "speed_kmh": 1050.0,
  "amount_z_score": 0.5,
  "amount_anomaly_flag": false,
  "recent_transactions_count": 2
}
"""

# 4. Añadimos la ID Card de Langfuse para la trazabilidad
@observe(name="Agent_MathExtractor", as_type="generation")
def extract_evidence(orchestrator_instructions: str, current_transaction: dict) -> dict:
    """
    Receives instructions from the Orchestrator and the current transaction data,
    uses the tools to get facts, and returns a JSON dictionary of evidence.
    """
    # Construimos el prompt dinámico
    user_prompt = f"""
    Instructions from Orchestrator: {orchestrator_instructions}
    
    Current Transaction Data:
    {json.dumps(current_transaction, indent=2)}
    
    Execute the necessary tools and return the JSON Evidence_Summary.
    """
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]
    
    # Invocamos al modelo con las herramientas
    response = llm_with_tools.invoke(messages)
    
    try:
        # Limpiamos posibles formatos extraños del LLM
        clean_json_str = response.content.strip().replace('```json', '').replace('```', '')
        return json.loads(clean_json_str)
    except json.JSONDecodeError:
        # Fallback de seguridad si el LLM no responde con JSON puro
        return {"error": "Failed to parse evidence", "raw_output": response.content}
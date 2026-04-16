# agents/extractor_4omini.py
import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langfuse import observe
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

# Importamos TUS herramientas matemáticas
from agents.tools import (
    calculate_distance, 
    check_impossible_travel, 
    calculate_amount_anomaly, 
    get_transactions_last_n_hours, 
    time_since_last_transaction
)

# 1. Definición de herramientas
tools = [
    calculate_distance, 
    check_impossible_travel, 
    calculate_amount_anomaly, 
    get_transactions_last_n_hours, 
    time_since_last_transaction
]

# 2. Configuración del modelo (Optimizado para Parallel Tool Calling)
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None,
    model="gpt-4o-mini", 
    temperature=0.0
)

# 3. SYSTEM PROMPT (Versión de Angel + Escapado de llaves {{ }})
SYSTEM_PROMPT = """
You are the strictly mathematical 'Evidence Extractor' agent for a fraud detection system.
Your ONLY job is to execute the specific analytical tools requested by the Orchestrator, 
read the mathematical results, and output a raw JSON summary.

### STRATEGIC DIRECTIVES:
1. PARALLELIZATION: To minimize latency, you MUST call multiple tools simultaneously whenever possible (e.g., trigger distance calculation AND amount anomaly at the exact same time).
2. IMPOSSIBLE TRAVEL LOGIC: Always evaluate physical distance vs time. If the speed required exceeds 900 km/h (plane speed), explicitly flag "is_physically_possible": false.
3. MERCHANT CONTEXTUALIZATION: Look at the transaction data. If the merchant relates to "Luxury", "Jewelry", "Casinos", "Crypto", or "Electronics", highlight this as an 'Aggravating Factor'.

RULES:
1. DO NOT guess or calculate numbers yourself. ALWAYS invoke the provided tools.
2. Your final output MUST be ONLY a valid JSON object. No markdown formatting, no conversational text.

Expected Output Format Example:
{{
  "agent_id": "math_extractor_4o_mini",
  "is_physically_possible": false,
  "distance_km": 1500.5,
  "amount_anomaly_flag": true,
  "merchant_risk_factor": "High Risk - Crypto Exchange",
  "summary": "Brief explanation of the extracted evidence."
}}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"), 
])

# 4. Agente y Ejecutor (Verbose=True para que veas la paralelización en consola)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@observe(name="Agent_MathExtractor", as_type="generation")
def extract_evidence(orchestrator_instructions: str, current_transaction: dict) -> dict:
    """
    Investiga la transacción usando herramientas y aplica el contexto de riesgo del comercio.
    """
    user_input = f"""
    Instructions from Orchestrator: {orchestrator_instructions}
    Current Transaction Data: {json.dumps(current_transaction, indent=2)}
    """
    
    # Invocamos al ejecutor
    response = agent_executor.invoke({"input": user_input})
    
    try:
        output_text = response["output"].strip()
        # Limpieza de bloques de código markdown
        if output_text.startswith("```"):
            output_text = output_text.split("\n", 1)[-1].rsplit("```", 1)[0]
            
        return json.loads(output_text.strip())
    except Exception as e:
        return {
            "error": "Failed to parse evidence JSON", 
            "details": str(e),
            "raw_output": response.get("output", "")
        }
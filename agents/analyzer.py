# agents/extractor_4omini.py
import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langfuse.decorators import observe
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Importamos TUS herramientas (Asumiendo que ya están hechas en agents/tools.py)
from agents.tools import (
    calculate_distance, 
    check_impossible_travel, 
    calculate_amount_anomaly, 
    get_transactions_last_n_hours, 
    time_since_last_transaction
)

tools = [
    calculate_distance, 
    check_impossible_travel, 
    calculate_amount_anomaly, 
    get_transactions_last_n_hours, 
    time_since_last_transaction
]

llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="gpt-4o-mini", 
    temperature=0.0
)

# 1. El prompt NECESITA un 'agent_scratchpad' para que el LLM recuerde qué herramientas ya usó
SYSTEM_PROMPT = """
You are the strictly mathematical 'Evidence Extractor' agent for a fraud detection system.
Your ONLY job is to execute the specific analytical tools requested by the Orchestrator, 
read the mathematical results, and output a raw JSON summary.

RULES:
1. DO NOT guess or calculate numbers yourself. ALWAYS invoke the provided tools.
2. Your final output MUST be ONLY a valid JSON object. No markdown formatting, no conversational text.

Expected Output Format Example:
{
  "agent_id": "math_extractor_4o_mini",
  "impossible_travel_detected": true,
  "distance_km": 1500.5,
  "amount_anomaly_flag": false
}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    # Este placeholder es vital: aquí LangChain insertará los resultados de las tools automáticamente
    ("placeholder", "{agent_scratchpad}"), 
])

# 2. Creamos el Agente oficial que sabe manejar "Tool Call IDs"
agent = create_tool_calling_agent(llm, tools, prompt)

# 3. El Executor es el "While Loop" que corre el código Python por el LLM
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@observe(name="Agent_MathExtractor", as_type="generation")
def extract_evidence(orchestrator_instructions: str, current_transaction: dict) -> dict:
    """
    Recibe instrucciones, usa las herramientas de Python automáticamente y devuelve el JSON.
    """
    user_input = f"""
    Instructions from Orchestrator: {orchestrator_instructions}
    Current Transaction Data: {json.dumps(current_transaction, indent=2)}
    """
    
    # Invocamos al ejecutor, no al LLM directamente
    response = agent_executor.invoke({"input": user_input})
    
    try:
        # La respuesta final en texto estará en la llave "output"
        clean_json_str = response["output"].strip().replace('```json', '').replace('```', '')
        return json.loads(clean_json_str)
    except json.JSONDecodeError:
        return {"error": "Failed to parse evidence", "raw_output": response.get("output", "")}
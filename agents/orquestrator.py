# agents/orchestrator.py
from langfuse.decorators import observe
from langfuse.langchain import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os

from agents.analyzer import extract_evidence

# Initialize your routing model (using the cheap one for routing to save the $40 budget)
router_model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="gpt-4o-mini", 
    temperature=0.0, # Keep it at 0 so it makes strict, deterministic routing decisions
)

@observe(name="Agent_Orchestrator", as_type="generation")
def run_orchestrator(transaction_data: dict, user_history: dict, session_id: str):
    """
    Acts as the router. Decides which specialized agent or tool to invoke based on transaction type.
    Returns the evidence_summary dictionary.
    """
    langfuse_handler = CallbackHandler()
    
    # 1. System Prompt directing the LLM on how to route (ALIGNED WITH PYTHON LOGIC)
    system_prompt = SystemMessage(content="""
    You are the Lead Fraud Orchestrator. 
    Analyze the transaction type and respond with ONLY one of the following exact routing commands:
    - 'ROUTE_TO_MATH_EXTRACTOR' if it is an 'in-person payment' (needs GPS distance) or if you need to calculate amount anomalies.
    - 'ROUTE_TO_TEXT_ANALYZER' if it is a 'bank transfer' or 'e-commerce' (needs to check SMS or Emails for phishing).
    - 'SAFE' if the transaction is obviously normal.
    """)
    
    user_prompt = HumanMessage(content=f"Transaction data: {transaction_data}")
    
    # 2. Invoke the model to get the routing decision
    response = router_model.invoke(
        [system_prompt, user_prompt], 
        config={
            "callbacks": [langfuse_handler],
            "metadata": {"langfuse_session_id": session_id},
        }
    )
    
    route_decision = response.content.strip()
    print(f"Orchestrator decided to: {route_decision}")
    
    # 3. Logic to actually call the next agents 
    if route_decision == "ROUTE_TO_MATH_EXTRACTOR":
        # Llamamos al extractor matemático que programó tu compañero
        evidence_json = extract_evidence(
            orchestrator_instructions="Calculate distance, velocity, and amount anomalies.",
            current_transaction=transaction_data
        )
        print(f"Evidencia matemática obtenida: {evidence_json}")
        return evidence_json

    elif route_decision == "ROUTE_TO_TEXT_ANALYZER":
        # TODO: Tu equipo de FinMath aún no ha creado las tools para leer correos/SMS.
        # Por ahora, devolvemos un JSON simulado para que el flujo no se rompa y el Juez pueda trabajar.
        print("Revisando textos (Simulación por ahora)...")
        simulated_evidence = {"text_analysis": "Pending implementation", "phishing_detected": False}
        return simulated_evidence
        
    else:
        # Si la ruta es 'SAFE' o algo que no entendió.
        return {"status": "safe_no_evidence_needed"}
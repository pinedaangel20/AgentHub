# agents/orchestrator.py
from langfuse import observe
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
    You are the Lead Fraud Orchestrator (Level 3 RiskOps).
    Your goal is to assess the 'Complexity Score' of a transaction and use 'Strategic Pruning' to route it efficiently.

    ### STRATEGIC PRUNING & ROUTING RULES:
    1. 'ROUTE_TO_MATH_EXTRACTOR': Use this if the transaction is an 'in-person payment' (needs GPS Haversine distance) OR if there is a massive Amount Anomaly. DO NOT trigger physical location checks for online transactions.
    2. 'ROUTE_TO_TEXT_ANALYZER': Use this if the transaction is 'bank transfer' or 'e-commerce' where checking SMS/Emails for phishing or Account Takeover (ATO) is the priority.
    3. 'SAFE': Use this ONLY if the Complexity Score is extremely low (normal amount, known device, reasonable time gap).

    ### COMPLEXITY SCORE ASSESSMENT (Internal Logic):
    - Deep Dive Needed: $0.00 or $1.00 followed by a huge amount, or sudden New Device.
    - Standard Check: Slight deviation in amount.
    - Low Risk: Matches historical baseline.

    ### GOLD STANDARD FRAUD EXAMPLES:
    - Example 1 (Card Testing): Tx1=$1.50, Tx2=$2000 in 5 mins -> 'ROUTE_TO_MATH_EXTRACTOR' (to verify velocity and Z-score).
    - Example 2 (Account Takeover): New Device, e-commerce, $500 -> 'ROUTE_TO_TEXT_ANALYZER' (to check emails for password resets).
    - Example 3 (Impossible Travel): In-person payment, same device, but moving 500km in 1 hour -> 'ROUTE_TO_MATH_EXTRACTOR' (to trigger GPS tools).

    OUTPUT FORMAT:
    You MUST output exactly ONE of the following strings and absolutely nothing else:
    ROUTE_TO_MATH_EXTRACTOR
    ROUTE_TO_TEXT_ANALYZER
    SAFE
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
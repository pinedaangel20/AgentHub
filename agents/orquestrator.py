# agents/orchestrator.py
import os
import json
from langfuse import observe
from langfuse.langchain import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Importación de agentes especializados
from agents.analyzer import extract_evidence
from agents.text_analyzer import analyze_communications

# Inicialización del modelo de ruteo
router_model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None,
    model="gpt-4o-mini", 
    temperature=0.0,
)

@observe(name="Agent_Orchestrator", as_type="generation")
def run_orchestrator(transaction_data: dict, user_history: dict, session_id: str):
    """
    Lead Orchestrator: Decide la estrategia de investigación principal,
    pero ejecuta una fusión de evidencias (Math + Text) si los datos están disponibles.
    """
    langfuse_handler = CallbackHandler()
    
    # 1. SYSTEM PROMPT: Define la estrategia de investigación
    system_prompt = SystemMessage(content="""
    You are the Lead Fraud Orchestrator (Level 3 RiskOps).
    Determine the primary investigation route based on the transaction type.

    ### ROUTING STRATEGIES:
    1. 'ROUTE_TO_MATH_EXTRACTOR': For 'in-person payment' or 'withdrawal'. Focus on GPS and patterns.
    2. 'ROUTE_TO_TEXT_ANALYZER': For 'bank transfer' or 'e-commerce'. Focus on Phishing and ATO.
    3. 'SAFE': For low-risk, routine transactions.

    OUTPUT FORMAT: Output ONLY the string: ROUTE_TO_MATH_EXTRACTOR, ROUTE_TO_TEXT_ANALYZER, or SAFE.
    """)
    
    user_prompt = HumanMessage(content=f"Transaction data: {json.dumps(transaction_data)}")
    
    # 2. Obtener decisión estratégica del Orquestador
    response = router_model.invoke(
        [system_prompt, user_prompt], 
        config={
            "callbacks": [langfuse_handler],
            "metadata": {"langfuse_session_id": session_id},
        }
    )
    
    route_decision = response.content.strip()
    print(f"🎯 Strategic Route Decision: {route_decision}")

    # Inicializamos contenedores
    math_evidence = {}
    text_evidence = {}

    # 3. EJECUCIÓN DINÁMICA (Fusión de Evidencias)
    
    # --- ANÁLISIS MATEMÁTICO ---
    # Se ejecuta si el Orquestador lo pide O si hay datos GPS/Montos para analizar
    if route_decision == "ROUTE_TO_MATH_EXTRACTOR" or transaction_data.get('lat'):
        print("🧮 Running Math & Pattern Analysis...")
        instructions = "Investigate GPS travel, home proximity, and structuring."
        math_evidence = extract_evidence(instructions, transaction_data)

    # --- ANÁLISIS DE TEXTO ---
    # Se ejecuta si el Orquestador lo pide O si hay comunicaciones presentes
    recent_comms = transaction_data.get('recent_communications', [])
    if route_decision == "ROUTE_TO_TEXT_ANALYZER" or recent_comms:
        if recent_comms:
            print(f"💬 Analyzing {len(recent_comms)} communications for Social Engineering...")
            text_evidence = analyze_communications(transaction_data.get('user_id'), recent_comms)
        else:
            text_evidence = {"status": "No text evidence found for analysis."}

    # 4. REPORTE UNIFICADO PARA EL JUEZ
    return {
        "metadata": {
            "primary_route": route_decision,
            "transaction_type": transaction_data.get('transaction_type'),
            "session_id": session_id
        },
        "math_evidence": math_evidence,
        "text_evidence": text_evidence
    }
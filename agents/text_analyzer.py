# agents/text_analyzer.py
import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langfuse import observe

# Configuración del modelo
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None,
    model="gpt-4o-mini",
    temperature=0.0
)

# El "Cerebro" del Agente
SYSTEM_PROMPT = """
You are a Senior Cybersecurity Analyst specializing in Social Engineering.
Your goal is to analyze SMS and Email logs to protect user {user_id}.

DETECTION CATEGORIES:
1. PHISHING: Links to suspicious domains or requests for sensitive data.
2. URGENCY: Phrases like "account blocked", "verify immediately", or "urgent action".
3. ACCOUNT TAKEOVER (ATO): Recent password reset requests or 2FA/OTP codes.
4. IBAN/PIVOT: Requests to change bank details or send money to a new "safe" account.

OUTPUT RULES:
- Return ONLY a valid JSON object.
- Use these exact keys: "phishing_detected", "urgency_detected", "risk_level", "reason".
"""

@observe(name="Agent_TextAnalyzer")
def analyze_communications(user_id: str, communications: list) -> dict:
    """
    Analiza SMS y Emails buscando patrones de fraude e Ingeniería Social.
    """
    if not communications:
        return {
            "agent_id": "text_behavior_analyzer",
            "social_engineering_detected": False,
            "phishing_risk": "Low",
            "reason": "No communications available for analysis."
        }

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Analyze these logs for user {user_id}: {comms}")
    ])
    
    chain = prompt_template | llm
    
    response = chain.invoke({
        "user_id": user_id,
        "comms": json.dumps(communications)
    })
    
    try:
        # 1. Limpieza del contenido
        raw_content = response.content.strip()
        if raw_content.startswith("```"):
            raw_content = raw_content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            
        result = json.loads(raw_content)

        # 2. SINCRONIZACIÓN DE LLAVES (Vital para el Juez y el Test)
        # Forzamos que el resultado tenga la llave que el resto del sistema espera.
        is_social_eng = result.get("phishing_detected", False) or result.get("urgency_detected", False)
        
        return {
            "agent_id": "text_behavior_analyzer",
            "social_engineering_detected": bool(is_social_eng),
            "phishing_risk": result.get("risk_level", "High" if is_social_eng else "Low"),
            "reason": result.get("reason", "Analysis completed.")
        }

    except Exception as e:
        # Fallback seguro en caso de error de parsing (Ya no causará NameError)
        return {
            "agent_id": "text_behavior_analyzer",
            "social_engineering_detected": True, # Bloqueamos por precaución si el texto es tan raro que rompe el JSON
            "phishing_risk": "High",
            "reason": f"Error parsing AI response: {str(e)}. Raw: {response.content[:50]}"
        }
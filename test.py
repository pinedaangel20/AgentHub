# test.py
import pandas as pd
import json
import os
import ulid
from dotenv import load_dotenv

# 1. CARGAR VARIABLES DE ENTORNO
load_dotenv()

from langfuse import Langfuse

# 2. IMPORTACIONES DE TU PROYECTO
from utils.preprocessor import preprocess_transactions
import agents.tools as math_tools
from agents.orquestrator import run_orchestrator
from agents.judge import judge_transaction

def run_e2e_test():
    print("🚀 === INICIANDO PRUEBA DE FUSIÓN TOTAL (MATH + TEXT + HOME) === 🚀\n")

    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: No se encontró API_KEY en tu .env.")
        return

    langfuse_client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
    )
    
    session_id = f"TEST-FULL-FRAUD-{ulid.new().str}"

    # ---------------------------------------------------------
    # FASE 1: DATA DE USUARIOS (NUEVO: Dirección de Casa)
    # ---------------------------------------------------------
    # Simulamos el perfil del usuario U-001 (Basado en Users.csv)
    math_tools.USER_PROFILES["U-001"] = {
        "home_lat": 52.52, # Berlín
        "home_lng": 13.40,
        "name": "Andres Quant"
    }

    # ---------------------------------------------------------
    # FASE 2: INGESTA DE DATOS (Structuring + Viaje Imposible)
    # ---------------------------------------------------------
    print("📊 [1/5] Creando transacciones simuladas...")
    data = {
        "transaction_id": ["T-01", "T-02", "T-03", "T-04"],
        "user_id": ["U-001", "U-001", "U-001", "U-001"],
        "amount": [10.0, 15.0, 12.0, 850.0],
        "timestamp": [
            "2026-04-16T14:00:00Z",
            "2026-04-16T14:05:00Z",
            "2026-04-16T14:10:00Z",
            "2026-04-16T14:15:00Z"
        ],
        "device_id": ["phone-A", "phone-A", "phone-A", "hacker-pc-X"],
        "lat": [52.52, 52.52, 52.52, 48.13], # Berlín -> Múnich
        "lng": [13.40, 13.40, 13.40, 11.58],
        # Cambiamos T-04 a 'bank transfer' para probar Phishing
        "transaction_type": ["in-person payment", "in-person payment", "in-person payment", "bank transfer"],
        "recipient_iban": [None, None, None, "DE891234567890"] 
    }
    df_input = pd.DataFrame(data)
    df_input["timestamp"] = pd.to_datetime(df_input["timestamp"], utc=True)

    # ---------------------------------------------------------
    # FASE 3: DATASTORE (Inyección de Memoria)
    # ---------------------------------------------------------
    print("💾 [2/5] Inyectando historial y perfiles...")
    for user_id, group in df_input.groupby("user_id"):
        math_tools.USER_DATA[user_id] = group.copy()
    
    # Preparamos la transacción T-04 para el análisis
    suspicious_row_df = df_input[df_input["transaction_id"] == "T-04"].copy()
    suspicious_row = json.loads(suspicious_row_df.to_json(orient="records", date_format="iso"))[0]

    # ---------------------------------------------------------
    # FASE 4: INYECCIÓN DE TEXTO (NUEVO: Phishing Evidence)
    # ---------------------------------------------------------
    # Añadimos los mensajes que el TextAnalyzer debe encontrar
    suspicious_row["recent_communications"] = [
        {"type": "SMS", "text": "Urgent: Your account is restricted. Verify your new IBAN here: http://bit.ly/secure-bank-login"},
        {"type": "Email", "text": "Security Alert: A new device has accessed your account from Munich."}
    ]

    # ---------------------------------------------------------
    # FASE 5: ORQUESTADOR Y JUEZ
    # ---------------------------------------------------------
    print(f"🧠 [3/5] Invocando Orquestador para {suspicious_row['transaction_id']}...")
    
    try:
        # El orquestador ahora devolverá un reporte de fusión (Math + Text)
        fusion_report = run_orchestrator(
            transaction_data=suspicious_row, 
            user_history={}, 
            session_id=session_id
        )
        
        print("\n🔍 REPORTE DE FUSIÓN (EVIDENCIA):")
        print(json.dumps(fusion_report, indent=2))
        
        print("\n⚖️  [4/5] Invocando al Juez Final...")
        decision = judge_transaction(fusion_report)
        
        print("\n🎯 ¡VEREDICTO FINAL DEL JUEZ! 🎯")
        print(json.dumps(decision, indent=4))
        
        # Validación del ÉXITO del test
        if decision.get("decision") == "BLOCK" and fusion_report.get("text_evidence", {}).get("social_engineering_detected"):
            print("\n🚨 ÉXITO TOTAL: El sistema detectó Phishing, Estructuración y Viaje Imposible.")
        else:
            print("\n⚠️ ALERTA: Revisa los logs. El sistema podría haber pasado por alto alguna evidencia.")

    except Exception as e:
        print(f"❌ Error en la ejecución: {e}")
        import traceback
        traceback.print_exc()

    langfuse_client.flush()
    print("\n📡 Prueba finalizada.")

if __name__ == "__main__":
    run_e2e_test()
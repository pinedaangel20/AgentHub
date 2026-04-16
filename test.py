# test.py
import pandas as pd
import json
import os
import ulid
from dotenv import load_dotenv

# 1. CARGAR VARIABLES DE ENTORNO ANTES QUE LAS IMPORTACIONES DE AGENTES
load_dotenv()

from langfuse import Langfuse

# 2. IMPORTACIONES DE TU PROYECTO
from utils.preprocessor import preprocess_transactions
import agents.tools as math_tools
from agents.orquestrator import run_orchestrator
from agents.judge import judge_transaction

def run_e2e_test():
    print("🚀 === INICIANDO PRUEBA END-TO-END (E2E) DEL SISTEMA MAS === 🚀\n")

    # Verificación de credenciales
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: No se encontró API_KEY en tu .env.")
        return

    langfuse_client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
    )
    
    session_id = f"TEST-{ulid.new().str}"

    # ---------------------------------------------------------
    # FASE 1: INGESTA DE DATOS
    # ---------------------------------------------------------
    print("📊 [1/5] Creando transacciones simuladas...")
    data = {
        "transaction_id": ["T-01", "T-02", "T-03", "T-04"],
        "user_id": ["U-001", "U-001", "U-001", "U-001"],
        "amount": [20.0, 25.0, 22.0, 850.0],
        "timestamp": [
            "2026-04-16T10:00:00Z",
            "2026-04-16T11:30:00Z",
            "2026-04-16T14:00:00Z",
            "2026-04-16T14:15:00Z"
        ],
        "device_id": ["phone-A", "phone-A", "phone-A", "hacker-pc-X"],
        "lat": [52.52, 52.52, 52.52, 48.13],
        "lng": [13.40, 13.40, 13.40, 11.58],
        "transaction_type": ["in-person payment"] * 4
    }
    df_input = pd.DataFrame(data)
    df_input["timestamp"] = pd.to_datetime(df_input["timestamp"], utc=True)

    # ---------------------------------------------------------
    # FASE 2: PREPROCESADOR
    # ---------------------------------------------------------
    print("⚙️  [2/5] Ejecutando Preprocesador de FinMath...")
    flagged_df = preprocess_transactions(df_input)
    
    if flagged_df.empty:
        print("ℹ️ No hay transacciones sospechosas para revisión.")
        return

    # ---------------------------------------------------------
    # FASE 3: DATASTORE (Inyección en RAM)
    # ---------------------------------------------------------
    print("💾 [3/5] Inyectando historial en USER_DATA (RAM)...")
    for user_id, group in df_input.groupby("user_id"):
        math_tools.USER_DATA[user_id] = group.copy()
    print("✅ Historial listo.\n")

    # --- LIMPIEZA ATÓMICA PARA EVITAR EL ERROR DE TIMESTAMP ---
    # Convertimos la fila a JSON y de vuelta a dict para asegurar tipos nativos de Python
    suspicious_row_raw = flagged_df.iloc[0:1].to_json(orient="records", date_format="iso")
    suspicious_row = json.loads(suspicious_row_raw)[0]
    # ----------------------------------------------------------

    # ---------------------------------------------------------
    # FASE 4: ORQUESTADOR
    # ---------------------------------------------------------
    print(f"🧠 [4/5] Invocando Orquestador para TX: {suspicious_row['transaction_id']}...")
    try:
        evidence_summary = run_orchestrator(
            transaction_data=suspicious_row, 
            user_history={}, 
            session_id=session_id
        )
        print("\n🔍 Evidencia extraída:")
        print(json.dumps(evidence_summary, indent=2))
        
    except Exception as e:
        print(f"❌ Error en Orquestador/Analyzer: {e}")
        return

    # ---------------------------------------------------------
    # FASE 5: JUEZ
    # ---------------------------------------------------------
    print("\n⚖️  [5/5] Invocando al Juez para el veredicto final...")
    try:
        judge_input = {
            "transaction_id": suspicious_row["transaction_id"],
            "amount": suspicious_row["amount"],
            "risk_context": suspicious_row.get("risk_context", "N/A"),
            "evidence_from_tools": evidence_summary 
        }
        decision = judge_transaction(judge_input)
        
        print("\n🎯 ¡VEREDICTO FINAL DEL JUEZ! 🎯")
        print(json.dumps(decision, indent=4))
        
    except Exception as e:
        print(f"❌ Error en el Juez: {e}")

    langfuse_client.flush()
    print("\n📡 Datos enviados a Langfuse. Prueba terminada.")

if __name__ == "__main__":
    run_e2e_test()
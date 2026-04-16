# test.py
import pandas as pd
import json
import os
import ulid
from dotenv import load_dotenv

# 1. CARGAR VARIABLES DE ENTORNO (Debe ir antes de importar los agentes)
load_dotenv()

from langfuse import Langfuse

# 2. IMPORTACIONES DE TU PROYECTO
from utils.preprocessor import preprocess_transactions
import agents.tools as math_tools
from agents.orquestrator import run_orchestrator
from agents.judge import judge_transaction

def run_e2e_test():
    print("🚀 === INICIANDO PRUEBA END-TO-END (E2E) CON MEMORIA Y PATRONES === 🚀\n")

    # Verificación de credenciales
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: No se encontró API_KEY en tu .env.")
        return

    langfuse_client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
    )
    
    session_id = f"TEST-PATTERN-{ulid.new().str}"

    # ---------------------------------------------------------
    # FASE 1: INGESTA DE DATOS (Simulando un ataque de "Structuring")
    # ---------------------------------------------------------
    print("📊 [1/5] Creando transacciones simuladas...")
    data = {
        "transaction_id": ["T-01", "T-02", "T-03", "T-04"],
        "user_id": ["U-001", "U-001", "U-001", "U-001"],
        "amount": [10.0, 15.0, 12.0, 850.0], # 3 pequeñas (Structuring) + 1 sospechosa
        "timestamp": [
            "2026-04-16T14:00:00Z",
            "2026-04-16T14:05:00Z",
            "2026-04-16T14:10:00Z",
            "2026-04-16T14:15:00Z"  # Solo 5 minutos entre cada una
        ],
        "device_id": ["phone-A", "phone-A", "phone-A", "hacker-pc-X"],
        "lat": [52.52, 52.52, 52.52, 48.13], # Berlín -> Berlín -> Berlín -> Múnich
        "lng": [13.40, 13.40, 13.40, 11.58],
        "transaction_type": ["in-person payment"] * 4
    }
    df_input = pd.DataFrame(data)
    
    # IMPORTANTE: Convertir a datetime para que las herramientas matemáticas no fallen
    df_input["timestamp"] = pd.to_datetime(df_input["timestamp"], utc=True)

    # ---------------------------------------------------------
    # FASE 2: PREPROCESADOR (Triaje de FinMath)
    # ---------------------------------------------------------
    print("⚙️  [2/5] Ejecutando Preprocesador de Alto Rendimiento...")
    flagged_df = preprocess_transactions(df_input)
    print(f"✅ Preprocesador analizó {len(df_input)} transacciones.\n")

    # ---------------------------------------------------------
    # FASE 3: DATASTORE (Inyección de MEMORIA en USER_DATA)
    # ---------------------------------------------------------
    print("💾 [3/5] Inyectando historial en la RAM de los Agentes...")
    for user_id, group in df_input.groupby("user_id"):
        math_tools.USER_DATA[user_id] = group.copy()
    print("✅ Memoria histórica establecida.\n")

    # --- SELECCIÓN DE LA TRANSACCIÓN SOSPECHOSA (T-04) ---
    # Usamos T-04 porque es la que tiene el patrón de Structuring y el Viaje Imposible
    suspicious_row_raw = df_input[df_input["transaction_id"] == "T-04"].to_json(orient="records", date_format="iso")
    suspicious_row = json.loads(suspicious_row_raw)[0]
    # ------------------------------------------------------

    # ---------------------------------------------------------
    # FASE 4: ORQUESTADOR Y ANALIZADOR DE PATRONES
    # ---------------------------------------------------------
    print(f"🧠 [4/5] Invocando Orquestador para investigar T-04...")
    print(f"   Monto: ${suspicious_row['amount']} | Ubicación: Múnich | Tiempo desde anterior: 5 min")
    
    try:
        # Aquí el Agente PatternAnalyzer usará las 6 herramientas en paralelo
        evidence_summary = run_orchestrator(
            transaction_data=suspicious_row, 
            user_history={}, 
            session_id=session_id
        )
        print("\n🔍 EVIDENCIA DETECTADA POR LOS AGENTES:")
        print(json.dumps(evidence_summary, indent=2))
        
    except Exception as e:
        print(f"❌ Error en el flujo de agentes: {e}")
        return

    # ---------------------------------------------------------
    # FASE 5: JUEZ (Veredicto Final)
    # ---------------------------------------------------------
    print("\n⚖️  [5/5] Invocando al Juez para decidir el bloqueo...")
    try:
        judge_input = {
            "transaction_id": suspicious_row["transaction_id"],
            "amount": suspicious_row["amount"],
            "risk_context": "High volume after small bursts", # Contexto del preprocesador
            "evidence_from_tools": evidence_summary 
        }
        decision = judge_transaction(judge_input)
        
        print("\n🎯 ¡VEREDICTO FINAL DEL JUEZ! 🎯")
        print(json.dumps(decision, indent=4))
        
        if decision.get("decision") == "BLOCK":
            print("\n🚨 ÉXITO: El sistema detectó el 'Structuring' y el 'Viaje Imposible'. Transacción BLOQUEADA.")
        else:
            print("\n⚠️ ALERTA: El sistema dejó pasar la transacción. Revisa la lógica del Juez.")
            
    except Exception as e:
        print(f"❌ Error en el Juez: {e}")

    langfuse_client.flush()
    print("\n📡 Prueba finalizada. Logs disponibles en Langfuse.")

if __name__ == "__main__":
    run_e2e_test()
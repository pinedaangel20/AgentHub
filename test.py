# test.py
import pandas as pd
import json
import os
import ulid
from dotenv import load_dotenv
from langfuse import Langfuse

# Importaciones exactas basadas en tus nombres de archivo
from utils.preprocessor import preprocess_transactions
import agents.tools as math_tools
from agents.orquestrator import run_orchestrator
from agents.judge import judge_transaction

# Cargar variables de entorno
load_dotenv()

def run_e2e_test():
    print("🚀 === INICIANDO PRUEBA END-TO-END (E2E) DEL SISTEMA MAS === 🚀\n")

    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ ERROR: No se encontró OPENROUTER_API_KEY en tu .env.")
        return

    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
         print("❌ ERROR: Faltan credenciales de Langfuse en tu .env.")
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
    print("📊 [1/5] Creando transacciones simuladas (Berlín -> Múnich)...")
    
    data = {
        "transaction_id": ["T-01", "T-02", "T-03", "T-04"],
        "user_id": ["U-001", "U-001", "U-001", "U-001"],
        "amount": [20.0, 25.0, 22.0, 850.0], # Gasto final muy anómalo
        "timestamp": [
            "2026-04-16T10:00:00Z",
            "2026-04-16T11:30:00Z",
            "2026-04-16T14:00:00Z",
            "2026-04-16T14:15:00Z"  # 15 minutos de diferencia
        ],
        "device_id": ["phone-A", "phone-A", "phone-A", "hacker-pc-X"],
        "lat": [52.52, 52.52, 52.52, 48.13], # Berlín a Múnich
        "lng": [13.40, 13.40, 13.40, 11.58],
        # Agregamos transaction_type para que el orquestador sepa rutear
        "transaction_type": ["in-person payment", "in-person payment", "in-person payment", "in-person payment"] 
    }
    df_input = pd.DataFrame(data)

    # ---------------------------------------------------------
    # FASE 2: PREPROCESADOR
    # ---------------------------------------------------------
    print("⚙️  [2/5] Ejecutando Preprocesador de FinMath...")
    flagged_df = preprocess_transactions(df_input)
    print(f"✅ {len(flagged_df)} transacción(es) marcada(s) para revisión LLM.\n")

    if flagged_df.empty:
        print("ℹ️ No hay transacciones sospechosas. Fin de la prueba.")
        return

    # ---------------------------------------------------------
    # FASE 3: DATASTORE (TOOLS)
    # ---------------------------------------------------------
    print("💾 [3/5] Inyectando historial en USER_DATA (RAM)...")
    math_tools.USER_DATA = dict(tuple(df_input.groupby("user_id")))
    print("✅ Historial listo para cálculos matemáticos.\n")

    # Tomamos la transacción sospechosa
    suspicious_row = flagged_df.iloc[0].to_dict()

    # ---------------------------------------------------------
    # FASE 4: ORQUESTADOR (Y ANALYZER)
    # ---------------------------------------------------------
    print("🧠 [4/5] Invocando Orquestador (Claude 3.5 Sonnet / gpt-4o-mini)...")
    print(f"   Revisando TX: {suspicious_row['transaction_id']} | Monto: ${suspicious_row['amount']}")
    
    try:
        # El Orquestador decidirá enviarlo al Math Extractor (analyzer.py) automáticamente
        evidence_summary = run_orchestrator(
            transaction_data=suspicious_row, 
            user_history={}, 
            session_id=session_id
        )
        print("\n🔍 Evidencia extraída por los agentes:")
        print(json.dumps(evidence_summary, indent=2))
        print("")
        
    except Exception as e:
        print(f"❌ Error en Orquestador/Analyzer: {e}")
        return

    # ---------------------------------------------------------
    # FASE 5: JUEZ (o3)
    # ---------------------------------------------------------
    print("⚖️  [5/5] Invocando al Juez (o3) para el veredicto final...")
    
    judge_features = {
        "transaction_id": suspicious_row["transaction_id"],
        "amount": suspicious_row["amount"],
        "risk_context": suspicious_row.get("risk_context", "N/A"),
        "evidence_from_tools": evidence_summary 
    }
    
    try:
        decision_json = judge_transaction(judge_features)
        
        print("\n🎯 ¡VEREDICTO FINAL DEL JUEZ! 🎯")
        print(json.dumps(decision_json, indent=4))
        
        if decision_json.get("decision") == "BLOCK":
            print("\n🚨 ÉXITO: ¡El Juez aplicó el costo asimétrico y bloqueó el fraude!")
        else:
            print("\n✅ ÉXITO: El sistema funcionó, pero el Juez decidió APROBARLA.")
            
    except Exception as e:
        print(f"❌ Error en el Juez: {e}")

    # Enviar datos a Langfuse
    langfuse_client.flush()
    print("\n📡 Datos enviados a Langfuse exitosamente.")

if __name__ == "__main__":
    run_e2e_test()
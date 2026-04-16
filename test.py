# test.py
import pandas as pd
import json
import os
from dotenv import load_dotenv

# Importar los módulos de tu equipo
from utils.preprocessor import preprocess_transactions
from agents.tools import USER_DATA
from agents.extractor_4omini import extract_evidence

# Cargar variables de entorno (Asegúrate de tener OPENAI_API_KEY en tu archivo .env)
load_dotenv()

def run_integration_test():
    print("🚀 === INICIANDO PRUEBA DE INTEGRACIÓN DEL SISTEMA === 🚀\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: No se encontró OPENAI_API_KEY. Asegúrate de tener tu archivo .env configurado.")
        return

    # ---------------------------------------------------------
    # FASE 1: INGESTA DE DATOS (Simulando el CSV real)
    # ---------------------------------------------------------
    print("📊 [1/4] Creando base de datos de transacciones simuladas...")
    
    # Creamos un usuario con transacciones normales en Berlín y un fraude repentino en Múnich
    data = {
        "user_id": ["U-001", "U-001", "U-001", "U-001"],
        "amount": [20.0, 25.0, 22.0, 850.0], # ¡El último gasto es de $850!
        "timestamp": [
            "2026-04-16T10:00:00Z",
            "2026-04-16T11:30:00Z",
            "2026-04-16T14:00:00Z",
            "2026-04-16T14:15:00Z"  # Solo 15 minutos después del gasto anterior
        ],
        "device_id": ["phone-A", "phone-A", "phone-A", "hacker-pc-X"], # Dispositivo nuevo
        "lat": [52.52, 52.52, 52.52, 48.13], # Berlín -> Berlín -> Berlín -> Múnich
        "lng": [13.40, 13.40, 13.40, 11.58]
    }
    df_input = pd.DataFrame(data)

    # ---------------------------------------------------------
    # FASE 2: EL PREPROCESADOR (El código de FinMath 1)
    # ---------------------------------------------------------
    print("⚙️  [2/4] Ejecutando el Preprocesador de alto rendimiento...")
    try:
        flagged_df = preprocess_transactions(df_input)
        print(f"✅ Preprocesador analizó {len(df_input)} transacciones y marcó {len(flagged_df)} para revisión LLM.\n")
    except Exception as e:
        print(f"❌ Error en el preprocesador: {e}")
        return

    # ---------------------------------------------------------
    # FASE 3: EL PUENTE (Guardar en RAM para las Matemáticas)
    # ---------------------------------------------------------
    print("💾 [3/4] Inyectando el historial del usuario en el DataStore (USER_DATA) de las tools...")
    # Agrupamos por usuario y lo metemos en tu diccionario global
    for user, df_user in df_input.groupby("user_id"):
        USER_DATA[user] = df_user.copy()
    print("✅ Historial inyectado en RAM para acceso instantáneo de Pandas.\n")

    # ---------------------------------------------------------
    # FASE 4: EL EXTRACTOR (Tu código de LLM)
    # ---------------------------------------------------------
    print("🤖 [4/4] Invocando a GPT-4o-mini para investigar la anomalía...")
    
    # Tomamos la primera transacción que el preprocesador marcó como "llm_review"
    suspicious_row = flagged_df.iloc[0].to_dict()
    
    # Simulamos la instrucción que el Orquestador le daría a tu Agente
    orchestrator_command = (
        "This transaction looks highly suspicious. "
        "1. Check the amount anomaly using the tools. "
        "2. The previous transaction was at lat 52.52, lng 13.40 (Berlin). "
        "The current is at lat 48.13, lng 11.58 (Munich). Check for impossible travel. "
        "Return the JSON Evidence Summary."
    )

    print("\n--- Analizando ---")
    print(f"Monto: ${suspicious_row['amount']} | Dispositivo: {suspicious_row['device_id']}")
    print(f"Contexto Matemático Previo (Z-Score): {suspicious_row['risk_context']}")
    print("------------------\n")

    try:
        # Llamamos a tu agente
        evidence_json = extract_evidence(orchestrator_command, suspicious_row)
        
        print("🎯 ¡RESULTADO FINAL DEL EXTRACTOR (JSON)! 🎯")
        print(json.dumps(evidence_json, indent=4))
        
        if evidence_json.get("impossible_travel_detected") == True:
            print("\n🚨 ÉXITO: ¡El LLM utilizó tu fórmula de Haversine y detectó el viaje imposible!")
        if evidence_json.get("amount_z_score", 0) > 3.0:
            print("🚨 ÉXITO: ¡El LLM detectó el gasto anómalo usando el Z-Score de Pandas!")
            
    except Exception as e:
        print(f"❌ Error en el Extractor: {e}")

if __name__ == "__main__":
    run_integration_test()
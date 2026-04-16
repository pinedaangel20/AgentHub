# main.py
import os
import pandas as pd
import json
from dotenv import load_dotenv
import ulid
from langfuse import Langfuse

# 1. Importaciones de módulos
from utils.preprocessor import preprocess_transactions
from agents.judge import judge_transaction
from agents.orquestrator import run_orchestrator
from utils.output_formatter import generate_submission_file, zip_project_for_submission
import agents.tools as math_tools

load_dotenv()

# Inicializar Langfuse (Asegúrate de que estas keys estén en tu .env)
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
)

def load_all_datasets():
    """Carga y cruza los archivos reales desde .data/evaluation/"""
    print("📂 Accediendo a la base de datos de Reply Mirror (.data/evaluation/)...")
    base_path = ".data/evaluation/"
    
    # --- 1. CARGAR TRANSACCIONES (CSV) ---
    df_tx = pd.read_csv(os.path.join(base_path, "transactions.csv"))
    df_tx["timestamp"] = pd.to_datetime(df_tx["timestamp"], utc=True)
    print(f"✅ Transacciones cargadas: {len(df_tx)}")
    
    # --- 2. CARGAR PERFILES DE USUARIO (JSON) ---
    try:
        # pd.read_json maneja automáticamente el parsing de JSON a DataFrame
        df_users = pd.read_json(os.path.join(base_path, "users.json"))
        # Poblamos el diccionario de tools.py para que el agente sepa dónde vive cada usuario [cite: 81]
        math_tools.USER_PROFILES = df_users.set_index("user_id").to_dict(orient="index")
        print(f"✅ Perfiles de usuario inyectados: {len(df_users)}")
    except Exception as e:
        print(f"⚠️ Error cargando users.json: {e}")

    # --- 3. CARGAR COMUNICACIONES (JSON) ---
    comms_list = []
    try:
        with open(os.path.join(base_path, "sms.json"), 'r') as f:
            sms_data = json.load(f) # Parser estándar de JSON para hilos de texto [cite: 82-84]
        with open(os.path.join(base_path, "mails.json"), 'r') as f:
            mail_data = json.load(f) # [cite: 85-86]
            
        # Unificamos ambos para el TextAnalyzer
        comms_list = sms_data + mail_data
        print(f"✅ Total comunicaciones para análisis: {len(comms_list)}")
    except Exception as e:
        print(f"⚠️ Error cargando comunicaciones: {e}")

    return df_tx, comms_list

def main():
    # Generamos el ID de sesión oficial para el tablero de jueces
    team_name = os.getenv("TEAM_NAME", "THE-EYE")
    session_id = f"{team_name}-{ulid.new().str}"
    print(f"\n🚀 EJECUCIÓN OFICIAL DEL RETO | Session: {session_id}")
    print("-" * 50)

    # 1. Carga masiva de datos reales
    raw_df, all_comms = load_all_datasets()
    
    # 2. Preparar el historial matemático para las herramientas (Tools)
    math_tools.USER_DATA = dict(tuple(raw_df.groupby('user_id')))

    # 3. Preprocesamiento (Filtro Determinístico para ahorrar tokens)
    print("⚙️  Ejecutando Triage de alta velocidad...")
    suspect_df = preprocess_transactions(raw_df)
    sus_count = len(suspect_df)
    print(f"🎯 Se detectaron {sus_count} transacciones sospechosas que requieren revisión de agentes.")
    print("-" * 50)

    final_fraud_ids = []

    # 4. Análisis Profundo con Agentes (MAS)
    for idx, row in suspect_df.iterrows():
        tx_dict = row.to_dict()
        user_id = tx_dict['user_id']
        
        # Saneamiento de fechas para evitar errores de JSON serializable
        for k, v in tx_dict.items():
            if isinstance(v, pd.Timestamp):
                tx_dict[k] = v.isoformat()
        
        # Cruzar con comunicaciones del dataset real
        # Filtramos solo los mensajes de este usuario específico para no saturar al LLM
        user_comms = [c for c in all_comms if c.get('user_id') == user_id]
        tx_dict['recent_communications'] = user_comms[-5:] # Tomamos los últimos 5 mensajes para contexto

        print(f"[{idx+1}/{sus_count}] Investigando ID: {tx_dict['transaction_id']} (Usuario: {user_id})")
        
        try:
            # EL ORQUESTADOR: Activa el MathExtractor y el TextAnalyzer si es necesario
            evidence = run_orchestrator(tx_dict, {}, session_id)
            
            # EL JUEZ: Dicta el veredicto final basado en las reglas del Problem Statement [cite: 48-50]
            judgment = judge_transaction(evidence)
            
            if judgment.get("decision") == "BLOCK":
                print(f"   🚨 FRAUDE DETECTADO: {judgment['reasoning'][:60]}...")
                final_fraud_ids.append(tx_dict['transaction_id'])
            else:
                print(f"   ✅ Transacción Legítima (Score: {judgment['fraud_score']})")
                
        except Exception as e:
            print(f"   ❌ Error crítico en el análisis de esta transacción: {e}")
            # Estrategia de seguridad: Si el sistema falla, bloqueamos transacciones grandes (>1000)
            if tx_dict['amount'] > 1000:
                final_fraud_ids.append(tx_dict['transaction_id'])

    # 5. Generar archivos de entrega oficial [cite: 87-92]
    print("-" * 50)
    generate_submission_file(final_fraud_ids, "output.txt")
    zip_project_for_submission("submission.zip")
    
    # Enviar métricas finales a Langfuse
    langfuse_client.flush()
    print("\n🏁 PROCESO COMPLETADO. Sube 'output.txt' y 'submission.zip' a la plataforma.")

if __name__ == "__main__":
    main()
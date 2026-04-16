#main.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import ulid
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler

load_dotenv()

# 1. Importaciones de tus compañeros
from utils.preprocessor import preprocess_transactions
from agents.judge import judge_transaction
# 2. Importaciones tuyas
from agents.orquestrator import run_orchestrator
from utils.output_formatter import generate_submission_file, zip_project_for_submission

# Cargar variables de entorno

# Inicializar Langfuse
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
)

def generate_session_id():
    """Generates a unique session ID as required by the challenge: TEAM-ULID"""
    team = os.getenv("TEAM_NAME", "default-team").replace(" ", "-")
    return f"{team}-{ulid.new().str}"

def load_challenge_dataset(filepath="data/Transactions.csv") -> pd.DataFrame:
    """Carga el dataset. Usaremos datos simulados si el archivo no existe para poder probar."""
    try:
        df = pd.read_csv(filepath)
        print(f"LE CHECK: Dataset loaded successfully with {len(df)} rows.")
        return df
    except FileNotFoundError:
        print(f"WARNING: {filepath} not found. Generating mock data for testing.")
        # Generamos datos falsos con la estructura que necesita preprocessor.py
        data = {
            "transaction_id": ["tx-01", "tx-02", "tx-03", "tx-04"],
            "user_id": [1, 1, 2, 2],
            "amount": [20.0, 5000.0, 30.0, 5.0],
            "timestamp": pd.date_range("2023-01-01", periods=4, freq="5min", tz="UTC"),
            "device_id": ["d1", "d2", "d3", "d3"],
            "transaction_type": ["e-commerce", "bank transfer", "in-person payment", "e-commerce"],
        }
        return pd.DataFrame(data)

def main():
    # 1. Iniciar sesión en Langfuse
    session_id = generate_session_id()
    print(f"\nLE CHECK: Starting Evaluation Run | Session ID: {session_id}")
    print("-" * 50)

    # 2. Cargar Dataset Original
    raw_df = load_challenge_dataset()
    
    # IMPORTANTE: Poblamos la "base de datos en memoria" de tools.py para que funcione
    import agents.tools as math_tools
    math_tools.USER_DATA = dict(tuple(raw_df.groupby('user_id')))

    # 3. Preprocesamiento Determinístico (Filtra la basura, deja lo dudoso)
    print("LE CHECK: Running Preprocessor...")
    suspect_df = preprocess_transactions(raw_df)
    
    suspect_count = len(suspect_df)
    print(f"LE CHECK: Triage Complete: {suspect_count} transactions flagged for LLM review.")
    print("-" * 50)

    final_fraud_ids = []

    # 4. Bucle Principal de Evaluación
    # Convertimos el DataFrame a diccionarios para que el LLM lo entienda fácil
    suspect_transactions = suspect_df.to_dict(orient="records")

    for idx, tx in enumerate(suspect_transactions, 1):
        print(f"LE CHECK: Analyzing Transaction {idx}/{suspect_count} (User: {tx['user_id']}, Amt: ${tx['amount']:.2f})")
        
        # A) EL ORQUESTADOR DECIDE LA RUTA (Y llama a las Tools si es necesario)
        # Tu orquestador internamente llamará a extract_evidence() si decide que necesita matemáticas
        evidence_summary = run_orchestrator(
            transaction_data=tx, 
            user_history={}, # Ya no lo pasamos así porque tools.py usa USER_DATA globalmente
            session_id=session_id
        )
        
        # B) EL JUEZ TOMA LA DECISIÓN FINAL BASADA EN COSTO ASIMÉTRICO
        # Formateamos los "features" (características) exactamente como lo pidió tu compañero
        judge_features = {
            "transaction_id": tx.get("transaction_id", "unknown"),
            "amount": tx["amount"],
            "risk_context": tx.get("risk_context", "No context"),
            "evidence_from_tools": evidence_summary # Pasamos lo que el Orquestador/Extractor averiguó
        }
        
        try:
            decision_json = judge_transaction(judge_features)
            is_fraud = decision_json.get("decision") == "BLOCK"
            
            print(f"   => Decision: {decision_json['decision']} | Confidence: {decision_json['confidence_level']}")
            
            if is_fraud:
                # Guardamos el ID como lo pide el reto
                final_fraud_ids.append(tx.get("transaction_id", "unknown-tx-id"))
                
        except Exception as e:
            print(f"   => ERROR: Error judging transaction: {e}")
            # Si el modelo falla por límite de cuota o error JSON, asumimos fraude (Costo asimétrico)
            if tx["amount"] > 1000:
                 final_fraud_ids.append(tx.get("transaction_id", "unknown-tx-id"))

    # 5. Asegurar envío de métricas a Langfuse
    print("-" * 50)
    print("LE CHECK: Flushing data to Langfuse...")
    langfuse_client.flush()

    # 6. Generar Outputs (Output.txt y Submission.zip)
    print("LE CHECK: Generating submission files...")
    generate_submission_file(final_fraud_ids, filename="output.txt")
    zip_project_for_submission(output_zip_name="submission.zip")
    
    print("\nLE CHECK: Execution complete. Good luck, team!")

if __name__ == "__main__":
    main()
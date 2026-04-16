#main.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import ulid
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler

from utils.output_formatter import generate_submission_file, zip_project_for_submission
# TODO: Import Sasha's preprocessor once its finished
# from utils.preprocessor import ...
# TODO: Import orchestrator funcs
# from agents.orchestrator import run_orchestrator

load_dotenv()

model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=50,
)

langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
)

def generate_session_id():
    team = os.getenv("TEAM_NAME", "tutorial").replace(" ", "-")
    return f"{team}-{ulid.new().str}"


def invoke_langchain(model, prompt, langfuse_handler, session_id):
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages, config={
        "callbacks": [langfuse_handler],
        "metadata": {"langfuse_session_id": session_id},
    })
    return response.content


@observe()
def run_llm_call(session_id, model, prompt):
    langfuse_handler = CallbackHandler()
    return invoke_langchain(model, prompt, langfuse_handler, session_id)


def main():
    session_id = generate_session_id()
    print(f"CHECK: Starting run with Session ID: {session_id}")

    # 1. LOAD AND PREPROCESS DATA (Waiting on FinMath team)
    # raw_data = load_dataset("data/Transactions.csv")
    # suspect_transactions, user_history = clean_data(raw_data)
    
    # MOCK DATA for testing your loop
    suspect_transactions = [
        {"id": "tx-001", "type": "e-commerce", "amount": 5000},
        {"id": "tx-002", "type": "in-person payment", "amount": 150}
    ]
    user_history = {} # Mock history
    
    final_fraud_ids = []

    # 2. EVALUATION LOOP
    print(f"Processing {len(suspect_transactions)} suspect transactions...")
    for tx in suspect_transactions:
        # TODO: Call your orchestrator here once it's built
        # is_fraud = run_orchestrator(tx, user_history, session_id)
        
        # MOCK DECISION
        is_fraud = True if tx["amount"] > 1000 else False 
        
        if is_fraud:
            final_fraud_ids.append(tx["id"])

    # 3. ENSURE LANGFUSE RECEIVES ALL DATA
    langfuse_client.flush()

    # 4. GENERATE OUTPUTS
    generate_submission_file(final_fraud_ids)
    zip_project_for_submission()
    
    print("\nCHECK: Execution complete. Check Langfuse dashboard.")

if __name__ == "__main__":
    main()


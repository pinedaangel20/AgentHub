# main.py

# TODO: Import dotenv and load environment variables (.env).
# TODO: Import ulid to generate the unique session_id (format: TEAM-ULID without spaces).

# TODO: Create a function to read the dataset (CSV or JSON) using Pandas or the built-in csv library.

# TODO: Create Preprocessing function (group_by_user).
# Group all dataset transactions by 'user_id' into a dictionary or DataFrame for quick access.

# TODO: Create rule-based filter (rule_based_filter).
# Before spending tokens, pass all transactions through strict rules (e.g., negative amounts, obvious impossible distances).
# If it's obvious fraud, flag and save it directly.

# TODO: Main evaluation loop.
# Iterate over the transactions that passed the initial filter and send them to agents/orchestrator.py.

# TODO: Ensure to use langfuse_client.flush() after calls to not lose monitoring data.

# TODO: Generate the output.txt file.
# Format the final list of fraudulent transactions exactly as requested in the challenge's "problem statement".

# TODO: (OPTIONAL BUT RECOMMENDED FOR EVALUATION) 
# Create a small automated script here or a bash file that compresses the whole folder into a .zip 
# (excluding venv, .env, and pycache) to have it ready for platform upload.

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import ulid
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler

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
    questions = [
        "What is machine learning?",
        "Explain neural networks briefly.",
        "What is the difference between AI and ML?"
    ]

    session_id = generate_session_id()

    for i, question in enumerate(questions, 1):
        response = run_llm_call(session_id, model, question)
        print(f"[{i}/{len(questions)}] {question} -> {response[:60]}...")

    langfuse_client.flush()

    print(f"\n{len(questions)} traces sent | session: {session_id}")
    print("Check the Langfuse dashboard to verify (may take a few minutes to update).")


if __name__ == "__main__":
    main()
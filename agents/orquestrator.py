# agents/orchestrator.py

# TODO: Import Langfuse (@observe and CallbackHandler) and LangChain/OpenRouter libraries.

# TODO: Initialize the Langfuse client with environment credentials.
# Ensure you use the correct URL: https://challenges.reply.com/langfuse

# TODO: Configure OpenRouter models.
# 1. Cheap model (e.g., gpt-4o-mini) for fast decisions.
# 2. Advanced model (e.g., claude-3-opus or gpt-4o) for complex cases.

# TODO: Bind the tools from agents/tools.py to the LangChain model.

# TODO: Create the main Orchestrator function (decorated with Langfuse's @observe).
# This function must:
# 1. Receive a transaction (and user context).
# 2. Decide whether to use the cheap or expensive model.
# 3. Invoke the selected model passing the Langfuse CallbackHandler and session_id.
# 4. Return a structured verdict (e.g., boolean: True for fraud, False for normal).
def run_orchestrator(transaction_data, session_id):
    pass

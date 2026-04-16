# agents/orchestrator.py
from langfuse import observe
from langfuse.langchain import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os

# Initialize your routing model (using the cheap one for routing to save the $40 budget)
router_model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="gpt-4o-mini", 
    temperature=0.0, # Keep it at 0 so it makes strict, deterministic routing decisions
)

@observe()
def run_orchestrator(transaction_data, user_history, session_id):
    """
    Acts as the router. Decides which specialized agent or tool to invoke based on transaction type.
    """
    langfuse_handler = CallbackHandler()
    
    # 1. System Prompt directing the LLM on how to route
    system_prompt = SystemMessage(content="""
    You are the Lead Fraud Orchestrator. 
    Analyze the transaction type and respond with ONLY one of the following routing commands:
    - 'ROUTE_TO_GPS' if it is an 'in-person payment'.
    - 'ROUTE_TO_SMS' if it is a 'bank transfer'.
    - 'ROUTE_TO_EMAIL' if it is 'e-commerce'.
    """)
    
    user_prompt = HumanMessage(content=f"Transaction data: {transaction_data}")
    
    # 2. Invoke the model to get the routing decision
    response = router_model.invoke(
        [system_prompt, user_prompt], 
        config={
            "callbacks": [langfuse_handler],
            "metadata": {"langfuse_session_id": session_id},
        }
    )
    
    route_decision = response.content.strip()
    print(f"Orchestrator decided to: {route_decision}")
    
    # 3. Logic to actually call the next agents (Waiting on FinMath team to finish the tools)
    # if route_decision == "ROUTE_TO_GPS":
    #     return run_extractor_agent(transaction_data, use_gps_tool=True)
    # ...
    
    return False # Placeholder return
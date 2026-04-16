# agents/extractor_4omini.py
import json
import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langfuse import observe
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

# 1. IMPORT TOOLS (Including your new structuring tool)
from agents.tools import (
    calculate_distance, 
    check_impossible_travel, 
    calculate_amount_anomaly, 
    get_transactions_last_n_hours, 
    time_since_last_transaction,
    check_structuring_pattern
)

# 2. REGISTER TOOLS
tools = [
    calculate_distance, 
    check_impossible_travel, 
    calculate_amount_anomaly, 
    get_transactions_last_n_hours, 
    time_since_last_transaction,
    check_structuring_pattern
]

# 3. MODEL CONFIGURATION
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None,
    model="gpt-4o-mini", 
    temperature=0.0
)

# 4. ADVANCED SYSTEM PROMPT
# Strategic Directives updated to emphasize 'Memory' and 'Smurfing' patterns.
SYSTEM_PROMPT = """
You are the 'Fraud Pattern Recognition' Expert. Your goal is to detect sophisticated fraud, 
specifically 'Structuring' (splitting large sums into small transactions).

### STRATEGIC DIRECTIVES:
1. MEMORY & STRUCTURING: You MUST invoke the `check_structuring_pattern` tool for every investigation. 
   This tool checks the 24-hour aggregate volume vs the user's historical average.
2. PATTERN RECOGNITION: Analyze the results from `get_transactions_last_n_hours`. 
   If you see 3+ transactions in a very short window (e.g., under 30 mins), flag it as 'High Velocity'.
3. PHYSICAL LOGIC: If 'time_since_last_transaction' is low but the distance is high, flag 'is_physically_possible': false.
4. PARALLELIZATION: Trigger distance, anomaly, and structuring checks simultaneously to reduce latency.

RULES:
1. ALWAYS output a valid JSON object. No conversational filler.
2. If `structuring_risk` from the tool is "High", you MUST report `structuring_detected`: true.

Expected Output Format Example:
{{
  "agent_id": "pattern_recognition_analyzer",
  "structuring_detected": true,
  "volume_vs_avg_ratio": 12.5,
  "transaction_velocity": "high",
  "is_physically_possible": false,
  "summary": "User split a large sum into 4 small payments. Total 24h volume is 12x their historical average."
}}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"), 
])

# 5. AGENT EXECUTOR
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@observe(name="Agent_PatternAnalyzer", as_type="generation")
def extract_evidence(orchestrator_instructions: str, current_transaction: dict) -> dict:
    """
    Analyzes behavior patterns and memory to detect structuring and anomalies.
    """
    # We explicitly tell the agent to check the patterns in the user_input
    user_input = f"""
    Orchestrator Instructions: {orchestrator_instructions}
    
    Target Transaction:
    {json.dumps(current_transaction, indent=2)}
    
    Please run a structuring check and velocity analysis for user {current_transaction.get('user_id')}.
    """
    
    # Simple brackets here (Python dict), not double brackets
    response = agent_executor.invoke({"input": user_input})
    
    try:
        output_text = response["output"].strip()
        # Clean Markdown formatting if present
        if output_text.startswith("```"):
            output_text = output_text.split("\n", 1)[-1].rsplit("```", 1)[0]
            
        return json.loads(output_text.strip())
    except Exception as e:
        # Fixed dictionary syntax (single brackets)
        return {
            "error": "JSON Parsing Error", 
            "details": str(e),
            "raw_output": response.get("output", "")
        }
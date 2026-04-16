# agents/judge_o3.py
# TODO: Import Langfuse (@observe). Verify the exact 'o3' model string in the Reply whitelisted models list.

# TODO: Define `make_final_judgment(evidence_summary, session_id)`.
# Decorate with @observe().

# TODO: Implement Sanity Check logic. 
# Prompt the Judge to return a JSON with {"is_fraud": true/false, "confidence_score": 0-100}.
# If confidence_score < 80, optionally trigger a loop to re-evaluate or default to marking as fraud to be safe.
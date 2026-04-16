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

if __name__ == "__main__":
    # Start the execution flow
    pass

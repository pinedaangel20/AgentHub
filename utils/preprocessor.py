# utils/preprocessor.py
import pandas as pd # or csv

# TODO: Create a function `clean_data(dataset)` to handle missing fields (e.g., fill None or drop corrupted rows).
# DO NOT use an LLM for this. Use pure Python.

# TODO: Create `rule_based_filter(transactions)`.
# Implement simple heuristics (e.g., "if amount < 0, flag as error", "if purchase is from same device and IP within 1 min, mark safe").
# Return only a list of `suspect_transactions` that actually need LLM cognitive effort.

# TODO: Create `group_by_user(transactions)` to return a dictionary mapping user_ids to their historical transactions.
# Reply AI Agent Challenge 2026 - Team Repository

Welcome to our team's repository for the Reply AI Agent Challenge. This project contains the agentic system required for the evaluation datasets.

## 📂 Files Overview

- **`main.py`**: The core script containing our AI agent's logic and Langchain/Langfuse integrations.
- **`.gitignore`**: Specifies intentionally untracked files to ignore (e.g., `venv/`, `.env`, `__pycache__/`) to prevent accidental uploads of secret keys or local cached files.
- **`.env.example`**: A template file with empty environment variables. Every team member must copy this, rename it to `.env`, and fill in their local keys.
- **`requirements.txt`**: A list of all necessary Python libraries to ensure reproducibility.
- **`README.md`**: This documentation file.

## ⚠️ Prerequisites

1.  **Python 3.13**: Highly recommended by the challenge guidelines. **Do not use Python 3.14** as it causes compatibility issues with Langfuse.
2.  **OpenRouter API Key**: Required for LLM access.
3.  **Langfuse Credentials**: Provided by the challenge platform.

## 💻 Installation & Setup Instructions

Follow the instructions below based on your Operating System to set up the project locally.

### For macOS Users

1.  **Clone the repository and navigate to the folder:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    ```
3.  **Activate the virtual environment:**
    ```bash
    source venv/bin/activate
    ```
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### For Windows Users

1.  **Clone the repository and navigate to the folder:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```
3.  **Activate the virtual environment:**
    ```bash
    venv\Scripts\activate
    ```
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🔐 Environment Variables Setup (All Users)

Once the dependencies are installed, you need to set up your local keys:

1.  Duplicate the `.env.example` file.
2.  Rename the duplicated file to exactly `.env`.
3.  Open `.env` and fill in your actual keys. _(Never commit this file to the repo!)_

## ▶️ Execution

To run the agent, simply execute:

```bash
python main.py
```

## 🤖 Multi-Agent System (MAS) Architecture

To maximize detection accuracy while strictly managing our API token budget (cost & latency), we implemented a hybrid architecture. It combines deterministic Python rules with a specialized LLM routing system.

### 🕵️‍♂️ Agent Roles

* **Rule-Based Preprocessor (Deterministic):** Not an LLM. A pure Python script that filters out "missing fields" and obviously safe/fraudulent transactions. It saves tokens by ensuring LLMs only process ambiguous cases.
* **Orchestrator Agent (Claude 3.5 Sonnet):** The system's "Brain". It receives suspect transactions from the preprocessor, analyzes the context, and dictates the investigation plan.
* **Evidence Extractor Agent (GPT-4o-mini):** The "Investigator". A cost-effective, fast model equipped with deterministic Python tools (Haversine distance, average spending calculators). It gathers mathematical evidence and creates a structured `evidence_summary`.
* **Fraud Judge Agent (o3 / High-Reasoning Model):** The "Executioner". It reviews the `evidence_summary`. It performs a sanity check based on a confidence threshold and delivers the final binary verdict (`Fraud: True/False`).

### 🌊 Execution Flow

1.  **Data Ingestion & Preprocessing:** `main.py` loads the dataset. `utils/preprocessor.py` cleans it and drops non-suspects.
2.  **Orchestration:** Suspect transactions are sent to `agents/orchestrator_sonnet.py`.
3.  **Tool Execution:** The Orchestrator delegates to `agents/extractor_4omini.py`, which triggers functions in `agents/tools.py` to calculate distances, time gaps, and anomalies.
4.  **Verdict:** The extracted evidence goes to `agents/judge_o3.py` for the final decision.
5.  **Output Generation:** `utils/output_formatter.py` collects all fraudulent IDs and writes them to the exact `.txt` format required for the evaluation submission, automatically generating the final `.zip` file.
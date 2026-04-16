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

# Agentic Document Chatbot with Multiple Sessions and Tools

This is a Streamlit-based chatbot application that allows users to upload DOCX or PDF documents and chat about their content using an AI agent powered by OpenAI's models and function calling. It supports multiple independent chat sessions, remembers the document uploaded per session, and can also answer general knowledge questions and provide the current time.

## Features

* **Document Understanding:** Upload DOCX or PDF files and ask questions about their content.
* **Multiple Chat Sessions:** Start new chat sessions to discuss different documents or topics independently. Each session maintains its own history and document context.
* **Agentic Capabilities:** The AI model acts as an agent, intelligently deciding when to use specific tools (like document search or getting the current time) to answer your questions.
* **Retrieval Augmented Generation (RAG):** Uses document embeddings and vector search (FAISS) to find relevant sections in your uploaded document to inform the AI's answers.
* **Tool Use:**
    * `search_document`: Searches the content of the uploaded document in the current session.
    * `get_current_datetime`: Provides the current date and time.
    * `summarize_section` (internal): Can be used by the agent to summarize retrieved document text.
* **General Knowledge:** Can answer questions that don't require document access or the current time.
* **Clear Session:** Easily clear the chat history and document associated with the current session.
* **Show Agent/Tool Steps:** Optional setting to visualize when the agent is deciding to use or execute a tool.

## Architecture

The application follows a modular architecture:

* **Streamlit UI:** Provides the interactive web interface.
* **Session Management:** Handles the state for multiple independent chat sessions using Streamlit's `st.session_state`.
* **Document Processing:** Extracts text from DOCX and PDF files and creates a searchable vector store (using `python-docx`, `PyMuPDF`, Langchain, and FAISS).
* **Tools:** Python functions representing external capabilities the AI can call.
* **Tool Definitions:** Metadata describing the tools to the OpenAI API.
* **Agent Logic:** Uses the OpenAI Chat Completions API with Function Calling to interpret user prompts, decide on actions (answer directly or call tools), execute tools, and generate final responses.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/PrashanthChinthapoola/Deep_Learning.git](https://github.com/PrashanthChinthapoola/Deep_Learning.git)
    cd Deep_Learning
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv .venv
    .venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your OpenAI API Key:**
    * Get an API key from the [OpenAI website](https://platform.openai.com/).
    * Create a file named `.env` in the root directory of the `Deep_Learning` folder (where `app.py` is located).
    * Add your API key to the `.env` file in this format:
        ```env
        OPENAI_API_KEY='your_openai_api_key_here'
        ```
    * **Important:** Ensure `.env` is in your `.gitignore` file to prevent accidentally committing your key. If you previously committed `.env` files (like `Ollama/.env` or `Agentic_app/.env`), you **must** remove them from your Git history using tools like `git filter-repo` or `git filter-branch` before pushing to a public repository.

## How to Run

Make sure your virtual environment is activated and you are in the root of the `Deep_Learning` directory.

```bash
streamlit run app.py
# Agentic Document Chatbot with Multiple Sessions and Tools

This is a Streamlit-based chatbot application that allows users to upload DOCX or PDF documents and chat about their content using an AI agent powered by OpenAI's models and function calling. It supports multiple independent chat sessions, remembers the document uploaded per session, and can also answer general knowledge questions and provide the current time.

## Features

* **Document Understanding (RAG):** Upload DOCX or PDF files. The application processes these documents by breaking them into smaller parts, creating numerical representations (embeddings), and storing them in a searchable database. When you ask a question about the document, the AI uses **Retrieval Augmented Generation (RAG)** to search this database for relevant information and uses it as context to formulate an accurate answer *based on the document's content*.
* **Multiple Chat Sessions:** Start new chat sessions using the sidebar. Each session is completely independent, meaning its chat history and the document you upload are specific to that session. This allows you to chat about different documents or topics simultaneously without mixing contexts.
* **Agentic Capabilities & Tool Use:** The AI model acts as a flexible agent. Instead of just generating text, it can **intelligently decide** if it needs to use an external **tool** to fulfill your request.
    * If you ask about the document, the agent calls the `search_document` tool to retrieve relevant text from the uploaded file's index.
    * If you ask for the current time, it calls the `get_current_datetime` tool.
    * It can also use an internal `summarize_section` tool if needed to condense long retrieved text.
    * For general questions, it answers directly using its built-in knowledge without needing tools.
    * The optional "Show Agent/Tool Steps" setting visualizes this decision-making and tool execution process.
* **General Knowledge:** Can answer questions that don't require document access or the current time, leveraging the base capabilities of the large language model.
* **Clear Session:** Easily clear the chat history, remove the uploaded document, and reset the context for the current session using the sidebar button.

## Architecture

The application follows a modular architecture that separates concerns for clarity and maintainability:

* **Streamlit UI:** Provides the interactive web interface the user sees and interacts with. Handles input fields, buttons, display areas for messages, and the sidebar layout.
* **Session Management (`session_manager.py`):** This module is crucial for maintaining the state of the application across reruns triggered by user interaction. It specifically manages the data for **multiple independent chat sessions**, storing the conversation history and the document-specific search index (`doc_retriever`) for each session in Streamlit's `st.session_state`.
* **Configuration (`config.py`):** Handles loading sensitive information like the `OPENAI_API_KEY` from environment variables (`.env` file). It also initializes global resources like the OpenAI client and the embeddings model, ensuring they are set up correctly at the application start.
* **Document Processing (`document_processor.py`):** This module contains the logic for preparing documents for search. It includes functions for extracting text from `.docx` and `.pdf` files (using `python-docx` and `PyMuPDF`). It then takes the extracted text, splits it into manageable chunks (`RecursiveCharacterTextSplitter`), converts these chunks into numerical vector embeddings (`OpenAIEmbeddings`), and builds a searchable index (`FAISS`) to enable efficient retrieval of relevant document snippets. Streamlit's caching decorators (`@st.cache_data`, `@st.cache_resource`) are used here to optimize performance by avoiding redundant processing.
* **Tools Implementation (`tools.py`):** This module contains the actual Python code that performs the actions callable by the AI agent. Each function here corresponds directly to a tool defined for the language model (e.g., the `search_document` function executes the vector search, `get_current_datetime_tool` gets the system time). These functions return results back to the agent.
* **Tool Definitions (`tool_definitions.py`):** This module serves as the bridge between the AI model and the Python tool implementations. It defines the tools in the specific JSON format required by the OpenAI API, including their names, descriptions (which the AI uses to understand when to call the tool), and expected parameters. It also maps these tool names to the corresponding Python functions in `tools.py` and holds the `system_prompt` that guides the agent's overall behavior and priorities (like prioritizing document search for relevant queries).
* **Agent Logic (`agent.py`):** This module embodies the core **"agentic" logic** and orchestrates the interaction with the OpenAI API, including the multi-step **Function Calling process**.
    * The main function, `process_chat_message(prompt: str)`, is called by `app.py` when the user submits input.
    * It prepares the full conversation history for the API by including the `system_prompt` and formatting messages (user, assistant, tool calls, tool outputs) correctly for the OpenAI API.
    * It makes the **first API call** (`openai_client.chat.completions.create`), sending the prompt, history, system prompt, and the list of `tools` from `tool_definitions.py`. The AI model processes this and decides its next action.
    * If the model's response includes `tool_calls`, this module **executes the tools**: It iterates through the requested `tool_calls`, uses the `available_functions` map to find the correct Python function, parses the arguments requested by the AI (`json.loads`), and runs the tool function (from `tools.py`).
    * It then prepares the messages for a **second API call**, including the original prompt, the model's decision to call tools, *and* the `content` returned by the tool executions (prefixed with "TOOL_RESPONSE:" or "TOOL_ERROR:").
    * It makes the second API call. This time, the AI model processes the tool results and synthesizes a **final, human-readable response** based on the tool's output and the original query.
    * If the model's initial response did *not* include `tool_calls`, the first response is treated as the final answer.
    * The module appends all intermediate API messages (tool calls, tool outputs) and the final assistant response to the current session's message history (`st.session_state.sessions[current_session_id]["messages"]`).
    * It handles API errors and tool execution errors, logging them and providing user-friendly messages in the chat.

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
    * **Crucial Secret Management:** **Never commit API keys or sensitive information directly to Git.** The `.env` file is used to keep your key separate from your code. Ensure the `.gitignore` file at the root of your repository contains lines like `.env` and `Agentic_app/.env` (and any other `.env` files) to prevent them from being tracked and pushed. If you have accidentally committed secrets in the past, you **must** remove them from your Git history using tools like `git filter-repo` or `git filter-branch` before pushing to a public repository.

## How to Run

Make sure your virtual environment is activated and you are in the root of the `Deep_Learning` directory.

```bash
streamlit run app.py
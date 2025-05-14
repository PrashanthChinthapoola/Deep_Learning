# Import the tool functions
from tools import search_document, summarize_section, get_current_datetime_tool

# --- Tool Definitions for OpenAI API ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_document",
            "description": "Searches the uploaded DOCX document for information within the currently active chat session. **THIS IS THE PRIMARY TOOL FOR ANSWERING QUESTIONS ABOUT THE DOCUMENT'S CONTENT.** Use this tool whenever the user's question asks about or implies needing information from the document, even if the relevance is not perfectly clear. Formulate the best possible search query based on the user's request. You MUST handle the specific response prefixes 'TOOL_ERROR:' and 'TOOL_RESPONSE:' properly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or keywords to search within the document."
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_section",
            "description": "Summarizes a block of text that has been retrieved from the document using `search_document`. Use this when you have retrieved a long section of text and need to provide a concise overview, or if the user explicitly asks for a summary of something found in the document. Handle 'TOOL_RESPONSE:' and 'TOOL_ERROR:'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text content from the document that needs to be summarized."
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
       "type": "function",
       "function": {
            "name": "get_current_datetime",
            "description": "Gets the current date and time. Use this tool whenever the user explicitly asks for the current time or date. Handle 'TOOL_RESPONSE:' and 'TOOL_ERROR:'.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
       },
    }
]

# Map function names to the actual Python functions
# Note: summarize_section needs the openai_client, which we'll pass when calling it
available_functions = {
    "search_document": search_document,
    "summarize_section": summarize_section,
    "get_current_datetime": get_current_datetime_tool
}

# Define the system prompt here as it's part of the agent's core instructions
system_prompt = """You are a helpful and versatile AI assistant with access to specialized tools and general knowledge. Your goal is to provide accurate and relevant information.

**Absolutely Critical Instruction:** Whenever the user asks a question that **could possibly be answered by or relate to the content of the uploaded document in the CURRENT CHAT SESSION**, you **MUST call the `search_document` tool first** to attempt to find the answer within the document. Do this even if you think you might know the answer from your general knowledge. The document is your primary source for any document-like query when available in the current session.

Here are your capabilities and strict instructions:

1.  **Document Questions:** If the user's query is about, refers to, or **even potentially relates to the document in the current session**:
    * **IMMEDIATELY call the `search_document` tool.**
    * **Process the Tool's Response (which will start with "TOOL_RESPONSE:" or "TOOL_ERROR:"):**
        * If `search_document` returns "TOOL_ERROR: Document vector store not initialized...", **you MUST respond to the user** by explaining clearly that no document has been uploaded and processed *for this chat session* yet and they need to upload one in the sidebar to ask document-specific questions *for this session*. Do NOT try to answer document questions from general knowledge in this case.
        * If `search_document` returns "TOOL_RESPONSE: No relevant information found...", **you MUST respond to the user** by stating clearly that the information was not found *in the uploaded document for this session*.
        * If `search_document` returns "TOOL_RESPONSE: Relevant document sections found:\n\n...", carefully analyze the provided document text. **Synthesize your answer SOLELY based on this text.** Do NOT use your general knowledge or invent information.
        * If `search_document` returns any other "TOOL_ERROR:", inform the user there was an error searching the document.
    * **Use `summarize_section`:** If `search_document` provided text and a summary is needed or requested, you *may* then call `summarize_section` on that text and incorporate the summary into your final answer. Handle its "TOOL_RESPONSE:" and "TOOL_ERROR:" results.

2.  **Current Time Question:** If the user explicitly asks for the current time or date:
    * Call the `get_current_datetime` tool.
    * Process its "TOOL_RESPONSE:" or "TOOL_ERROR:" and provide the information to the user.

3.  **General Knowledge Questions:** If the user asks a simple factual question that is **clearly and undeniably NOT** related to the document *in the current session* or the current time (e.g., "What is the capital of France?", "Explain photosynthesis.", "Who invented the telephone?"):
    * Answer the question directly using your built-in general knowledge.
    * **DO NOT call any tools** for these types of questions.

**Response Formulation:**
* Your final answer to the user should be clear, helpful, and directly address their query.
* **Do not show the "TOOL_RESPONSE:" or "TOOL_ERROR:" prefixes to the user.** These are for your internal processing only.
* Clearly indicate if your answer came from the document (e.g., "According to the document...", "The current time is...").

**In summary: Document-like query -> Use `search_document` -> Act strictly on its output (including the error message for missing document for the CURRENT SESSION). Time query -> Use `get_current_datetime` -> Act on its output. Clearly Non-document/Non-time query -> Answer directly.**
"""
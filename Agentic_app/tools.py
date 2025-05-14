import streamlit as st
import time
from datetime import datetime
import json # Needed for potential tool arg parsing if not done by agent
# Assuming openai_client is initialized in config and accessible
# from config import openai_client # Or pass it as argument
from session_manager import get_current_session # Import the function to get session state

def search_document(query: str) -> str:
    """
    Searches the uploaded DOCX document using the vector store of the CURRENT SESSION.
    Returns relevant document chunks as a string.
    Returns a specific error message if the vector store is not initialized for the current session.
    """
    current_session = get_current_session()
    retriever = current_session.get("doc_retriever")
    session_id = current_session.get("name", "Current Session") # Get session name for logging

    if st.session_state.get("show_tool_messages"):
        st.info(f"[{session_id}] 🛠️ Calling tool: `search_document` with query: '{query}'")

    if not retriever:
        # This specific error message is handled by the agent's system prompt
        return "TOOL_ERROR: Document vector store not initialized for this session. Please upload a document in the sidebar."
    try:
        time.sleep(0.5) # Simulate tool latency
        relevant_docs = retriever.invoke(query)

        if not relevant_docs:
            if st.session_state.get("show_tool_messages"):
                st.info(f"[{session_id}] ✔️ `search_document` found no relevant sections.")
            # This specific response is handled by the agent's system prompt
            return "TOOL_RESPONSE: No relevant information found in the document for that query in this session."

        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
        if st.session_state.get("show_tool_messages"):
            st.info(f"[{session_id}] ✔️ `search_document` found {len(relevant_docs)} relevant section(s).")
        max_context_length = 5000 # Prevent sending excessively long context to the model
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n\n[... context truncated ...]"
            if st.session_state.get("show_tool_messages"):
                st.warning(f"[{session_id}] Context from document truncated to {max_context_length} characters.")

        return "TOOL_RESPONSE: Relevant document sections found:\n\n" + context

    except Exception as e:
        st.error(f"[{session_id}] Error during document search tool execution: {e}")
        # Return specific error prefix for agent processing
        return f"TOOL_ERROR: An error occurred while searching the document: {e}"

def summarize_section(text: str, openai_client) -> str:
    """Summarizes a given text section using the OpenAI API."""
    from session_manager import get_current_session_id # Import here to avoid circular dependency

    if not text or len(text.strip()) < 50:
        if st.session_state.get("show_tool_messages"):
            st.warning(f"[{get_current_session_id()}] Input text for summarization is too short.")
        return "TOOL_RESPONSE: Cannot summarize a very short or empty text section."

    if st.session_state.get("show_tool_messages"):
        st.info(f"[{get_current_session_id()}] 🛠️ Calling tool: `summarize_section`...")
    try:
        time.sleep(0.5) # Simulate tool latency
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo", # Use a smaller model for summarization if preferred
            messages=[
                {"role": "system", "content": "You are a helpful assistant skilled at summarizing text concisely and accurately."},
                {"role": "user", "content": f"Please summarize the following text:\n\n{text}"},
            ],
            max_tokens=300,
            temperature=0.7
        )
        summary = response.choices[0].message.content
        if st.session_state.get("show_tool_messages"):
             st.info(f"[{get_current_session_id()}] ✔️ `summarize_section` complete.")
        return "TOOL_RESPONSE: Summary:\n\n" + summary
    except Exception as e:
        st.error(f"[{get_current_session_id()}] Error during summarization tool execution: {e}")
        # Return specific error prefix for agent processing
        return f"TOOL_ERROR: An error occurred while trying to summarize the section: {e}"

def get_current_datetime_tool() -> str:
    """Gets the current date and time."""
    from session_manager import get_current_session_id # Import here to avoid circular dependency

    if st.session_state.get("show_tool_messages"):
        st.info(f"[{get_current_session_id()}] 🛠️ Calling tool: `get_current_datetime`...")
    try:
        now = datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S %A")
        if st.session_state.get("show_tool_messages"):
             st.info(f"[{get_current_session_id()}] ✔️ `get_current_datetime` complete: {formatted_datetime}")
        # Return specific response prefix for agent processing
        return f"TOOL_RESPONSE: Current datetime is {formatted_datetime}"
    except Exception as e:
        st.error(f"[{get_current_session_id()}] Error getting current datetime: {e}")
        # Return specific error prefix for agent processing
        return f"TOOL_ERROR: An error occurred while getting the current date and time: {e}"
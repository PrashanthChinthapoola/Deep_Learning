# session_manager.py

import streamlit as st

def initialize_session_state():
    """Initializes session state variables for multiple chat sessions."""
    if "sessions" not in st.session_state:
        st.session_state.sessions = {}
        st.session_state.session_counter = 0
        create_new_session()
        if "show_tool_messages" not in st.session_state:
            st.session_state.show_tool_messages = False

def create_new_session():
    """Creates a new chat session and sets it as the current one."""
    st.session_state.session_counter += 1
    new_session_id = f"Chat {st.session_state.session_counter}"
    while new_session_id in st.session_state.sessions:
         st.session_state.session_counter += 1
         new_session_id = f"Chat {st.session_state.session_counter}"

    st.session_state.sessions[new_session_id] = {
        # Updated initial message
        "messages": [{"role": "assistant", "content": "Hello! Upload one or more DOCX or PDF files in the sidebar to chat about their combined content, or ask about the current time or general knowledge."}],
        "doc_retriever": None,
        "uploaded_file_name": None, # Will store comma-separated string of names
        "uploaded_file_info": None, # Store list of (name, size) tuples for change detection
        "name": new_session_id
    }
    st.session_state.current_session_id = new_session_id

def clear_current_session():
    """Clears the chat history and document for the current session."""
    current_session_id = st.session_state.current_session_id
    st.session_state.sessions[current_session_id] = {
        # Updated clear message
        "messages": [{"role": "assistant", "content": "Chat and document(s) cleared for this session. Upload new DOCX or PDF files to get started."}],
        "doc_retriever": None,
        "uploaded_file_name": None,
        "uploaded_file_info": None,
        "name": current_session_id
    }

def get_current_session():
    """Returns the state dictionary for the currently active session."""
    return st.session_state.sessions[st.session_state.current_session_id]

def get_all_session_ids():
    """Returns a list of all session IDs."""
    return list(st.session_state.sessions.keys())

def set_current_session(session_id: str):
    """Sets the specified session ID as the current one."""
    if session_id in st.session_state.sessions:
        st.session_state.current_session_id = session_id
        st.rerun()
    else:
        st.warning(f"Session ID '{session_id}' not found.")

initialize_session_state()
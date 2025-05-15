import streamlit as st

# Import modules
import config # Initializes OpenAI client and embeddings model
import session_manager # Manages sessions in st.session_state
import ui # Handles sidebar and chat rendering, including file upload trigger
import agent # Contains the agent logic for processing messages

# --- Main App Execution ---

st.set_page_config(page_title="Agentic Doc Chatbot + Time", layout="wide")
st.title("📄⏳ Agentic Chatbot for Your DOCX Document + Time")
st.caption("Upload a DOCX file in the sidebar to chat about it within the selected session. I can also tell you the current time and answer general knowledge questions.")


# Initialize sessions (this happens when session_manager is imported, but calling explicitly is clear)
session_manager.initialize_session_state()

# Render the sidebar
ui.render_sidebar()

# Render the chat area and get user prompt
prompt = ui.render_chat()

# Process the prompt if one was entered
if prompt:
    # The agent module handles the chat logic, including tool calls and API interactions
    agent.process_chat_message(prompt)

# Note: Streamlit handles reruns automatically when state changes or input is submitted.
# The file processing logic is now integrated into ui.render_sidebar to handle uploads
# and trigger reruns there.
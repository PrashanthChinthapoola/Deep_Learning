# ui.py

import streamlit as st
import document_processor
import session_manager
from config import embeddings_model # Need embeddings model for doc processing
# import sys # No longer needed for sys.path print


def render_sidebar():
    """Renders the Streamlit sidebar with session and upload controls."""
    with st.sidebar:
        st.title("Chat Options")

        st.subheader("Chat Sessions")

        session_ids = session_manager.get_all_session_ids()
        current_session_id = st.session_state.current_session_id # Get current session ID here

        for session_id in session_ids:
            is_current = (session_id == current_session_id)
            display_name = session_manager.st.session_state.sessions[session_id].get("name", session_id)
            button_label = f"➡️ **{display_name}**" if is_current else display_name

            if st.button(button_label, key=f"session_button_{session_id}", use_container_width=True):
                if not is_current:
                    session_manager.set_current_session(session_id)


        if st.button("➕ Start New Chat", use_container_width=True):
            session_manager.create_new_session()
            # Rerun is handled inside create_new_session


        st.markdown("---")

        st.write("") # Add some space
        if st.button(f"🗑️ Clear '{current_session_id}' Chat & Doc", use_container_width=True):
            session_manager.clear_current_session()
            st.rerun() # Rerun after clearing state


        st.markdown("---")

        st.subheader("Upload Document")
        current_session_state = session_manager.get_current_session()
        current_session_uploaded_file_name = current_session_state["uploaded_file_name"]

        uploaded_file = st.file_uploader(
            f"Choose DOCX or PDF for '{current_session_id}'", # Updated label
            type=["docx", "pdf"], # Allow both types
            help="Select a .docx or .pdf file to process for the currently selected chat session.", # Updated help text
            key=f"uploader_session_{current_session_id}" # Unique key per session
        )

        # File processing logic triggered by the uploader
        if uploaded_file is not None:
             uploaded_file_content = uploaded_file.getvalue()
             uploaded_file_name = uploaded_file.name

             # Check if this file is already processed for this session
             if current_session_state["uploaded_file_name"] != uploaded_file_name or current_session_state["doc_retriever"] is None:

                 # Add logging outside the cached function
                 if st.session_state.show_tool_messages:
                       session_manager.get_current_session()["messages"].append({"role": "assistant", "content": f"[{current_session_id}] Processing '{uploaded_file_name}'..."})


                 with st.spinner(f"Processing '{uploaded_file_name}' for '{current_session_id}'..."):
                     doc_text = None
                     if uploaded_file_name.lower().endswith('.docx'):
                          # Call cached function for DOCX
                         doc_text = document_processor.extract_text_from_docx(uploaded_file_content)
                     elif uploaded_file_name.lower().endswith('.pdf'):
                          # Call cached function for PDF
                         doc_text = document_processor.extract_text_from_pdf(uploaded_file_content)
                     else:
                          st.warning(f"[{current_session_id}] Unsupported file type: {uploaded_file_name}. Please upload a .docx or .pdf file.")
                          session_manager.get_current_session()["messages"].append({"role": "assistant", "content": f"[{current_session_id}] ⚠️ Unsupported file type: {uploaded_file_name}. Please upload a .docx or .pdf file."})


                     if doc_text:
                         # Call cached function to create vector store from extracted text
                         # Pass embeddings_model explicitly, prefixed with _ in function definition
                         retriever = document_processor.create_vector_store(doc_text, embeddings_model) # Still called with 2 args here

                         # Update session state *after* processing
                         session_manager.get_current_session()["doc_retriever"] = retriever
                         session_manager.get_current_session()["uploaded_file_name"] = uploaded_file_name

                         # Add logging outside the cached function
                         if retriever:
                              if st.session_state.show_tool_messages:
                                   session_manager.get_current_session()["messages"].append({"role": "assistant", "content": f"[{current_session_id}] ✅ '{uploaded_file_name}' processed successfully! You can now ask questions about its content."})
                              else:
                                   session_manager.get_current_session()["messages"].append({"role": "assistant", "content": f"[{current_session_id}] ✅ '{uploaded_file_name}' processed!"})
                         else:
                             session_manager.get_current_session()["messages"].append({"role": "assistant", "content": f"[{current_session_id}] ⚠️ Failed to create a searchable index for '{uploaded_file_name}'. The document might be empty or have formatting issues."})
                     elif doc_text is not None: # Handle the case where extraction worked but resulted in empty text
                          session_manager.get_current_session()["messages"].append({"role": "assistant", "content": f"[{current_session_id}] ⚠️ '{uploaded_file_name}' seems to be empty or text extraction failed."})
                     # If doc_text is None because of an unsupported file type, the message is already added above.


                 st.rerun() # Rerun to update the main chat area with processing messages and enable chat


        st.markdown("---")

        with st.expander("Settings"):
            st.subheader("Display Settings")
            st.session_state.show_tool_messages = st.checkbox(
                "Show Agent/Tool Steps",
                value=st.session_state.show_tool_messages,
                key="show_tool_messages_checkbox",
                help="Show intermediate steps where the agent calls tools."
            )

        st.markdown("---")

        with st.expander("About"):
            st.write("This is an agentic chatbot that can process a DOCX or PDF document (per session) and answer questions based on its content using function calling.") # Updated About
            st.write("It can also answer general questions and provide the current time.")
            st.write("It uses Streamlit for the interface and OpenAI's GPT models with function calling.")
            st.write("Powered by OpenAI, Streamlit, and PyMuPDF.") # Mention PyMuPDF


def render_chat():
    """Renders the main chat message area."""
    current_session = session_manager.get_current_session()
    current_messages = current_session["messages"]
    session_id = current_session.get("name", "Current Session")
    uploaded_file_name = current_session.get("uploaded_file_name", "No document uploaded")

    # Display messages
    for message in current_messages:
        # We only display 'user' and 'assistant' roles in the main chat bubble area
        # Tool messages are logged via st.info/st.success/st.error if show_tool_messages is True
        if message["role"] != "tool":
            with st.chat_message(message["role"]):
                if message.get("content") is not None:
                    # Optionally add a specific message when tool_calls are present
                    # The model's message might be empty or just a placeholder before tool results
                    if st.session_state.get("show_tool_messages") and message["role"] == "assistant" and message.get("tool_calls"):
                        # Avoid showing "Thinking..." if there's already content
                        if message["content"].strip() == "":
                             st.markdown(f"[{session_id}] 🤖 **Thinking... (Tool Call)**")
                        else:
                             # If there's content AND tool calls, display content then thinking...
                             st.markdown(message["content"])
                             st.markdown(f"[{session_id}] 🤖 **Thinking... (Tool Call)**")
                    else:
                        st.markdown(message["content"])
                # Handle the case where an assistant message has tool_calls but no content yet (before tool results)
                elif st.session_state.get("show_tool_messages") and message["role"] == "assistant" and message.get("tool_calls"):
                     st.markdown(f"[{session_id}] 🤖 **Thinking... (Tool Call)**")


    # Chat input area
    # Update prompt text
    prompt = st.chat_input(
        f"Ask about the document ({uploaded_file_name}), current time, or general knowledge in '{session_id}'..."
    )

    return prompt # Return the prompt to be processed by the agent in app.py
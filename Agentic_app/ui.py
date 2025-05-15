# ui.py

import streamlit as st
import document_processor
import session_manager
from config import embeddings_model
# import sys # No longer needed


def render_sidebar():
    """Renders the Streamlit sidebar with session and upload controls."""
    with st.sidebar:
        st.title("Chat Options")

        st.subheader("Chat Sessions")

        session_ids = session_manager.get_all_session_ids()
        current_session_id = st.session_state.current_session_id

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

        st.subheader("Upload Document(s)") # Updated subheader
        current_session_state = session_manager.get_current_session()
        current_session_uploaded_file_name = current_session_state["uploaded_file_name"] # This will store combined name(s) now

        # --- Modified File Uploader ---
        uploaded_files = st.file_uploader(
            f"Choose DOCX or PDF files for '{current_session_id}'", # Updated label
            type=["docx", "pdf"], # Allow both types
            help="Select one or more .docx or .pdf files to process for the currently selected chat session. A single searchable index will be created for all selected files.", # Updated help text
            key=f"uploader_session_{current_session_id}", # Unique key per session
            accept_multiple_files=True # <<< Allow multiple files
        )

        # --- Modified File Processing Logic ---
        if uploaded_files is not None and len(uploaded_files) > 0:
            # Check if the list of uploaded files has changed or no retriever exists
            # We compare file names and sizes for a simple check
            current_file_info = [(f.name, f.size) for f in uploaded_files]
            previous_file_info = current_session_state.get("uploaded_file_info") # Store info to detect changes

            if current_file_info != previous_file_info or current_session_state["doc_retriever"] is None:

                 # Store info about the files currently being processed
                 current_session_state["uploaded_file_info"] = current_file_info

                 file_names = [f.name for f in uploaded_files]
                 display_file_names = ", ".join(file_names)
                 if len(display_file_names) > 100: # Shorten display if many files
                      display_file_names = display_file_names[:100] + "..."

                 if st.session_state.show_tool_messages:
                       session_manager.get_current_session()["messages"].append({"role": "assistant", "content": f"[{current_session_id}] Processing file(s): '{display_file_names}'..."})


                 all_text = ""
                 successful_files = []
                 failed_files = []

                 with st.spinner(f"Processing {len(uploaded_files)} file(s) for '{current_session_id}'..."):
                     for uploaded_file in uploaded_files:
                          uploaded_file_content = uploaded_file.getvalue()
                          uploaded_file_name = uploaded_file.name.lower() # Use lower case for extension check

                          file_text = None
                          if uploaded_file_name.endswith('.docx'):
                               file_text = document_processor.extract_text_from_docx(uploaded_file_content)
                          elif uploaded_file_name.endswith('.pdf'):
                               file_text = document_processor.extract_text_from_pdf(uploaded_file_content)
                          else:
                               st.warning(f"[{current_session_id}] Skipping unsupported file type: {uploaded_file.name}")
                               failed_files.append(uploaded_file.name)
                               continue # Skip to next file

                          if file_text:
                               all_text += file_text + "\n\n---\n\n" # Concatenate text with separator
                               successful_files.append(uploaded_file.name)
                          else:
                               st.warning(f"[{current_session_id}] Failed to extract text from {uploaded_file.name} or file was empty.")
                               failed_files.append(uploaded_file.name)


                 # Process the combined text if any files were successful
                 retriever = None
                 if all_text.strip():
                      # Pass the combined text to create vector store
                      retriever = document_processor.create_vector_store(all_text, embeddings_model) # Still called with 2 args here

                      # Update session state with the single retriever
                      session_manager.get_current_session()["doc_retriever"] = retriever
                      session_manager.get_current_session()["uploaded_file_name"] = ", ".join(successful_files) # Store names of successful files


                 # Add processing result messages
                 status_messages = []
                 if successful_files:
                      success_msg = f"✅ Processed {len(successful_files)} file(s): {', '.join(successful_files)} for '{current_session_id}'!"
                      if retriever:
                           success_msg += " You can now ask questions about their combined content."
                      else:
                           success_msg += " But failed to create a searchable index from the text."
                      status_messages.append(success_msg)

                 if failed_files:
                      fail_msg = f"⚠️ Failed to process {len(failed_files)} file(s): {', '.join(failed_files)}."
                      status_messages.append(fail_msg)

                 if not successful_files and not failed_files:
                     status_messages.append(f"⚠️ No supported files selected or processed.")


                 for msg in status_messages:
                      if st.session_state.show_tool_messages:
                            session_manager.get_current_session()["messages"].append({"role": "assistant", "content": f"[{current_session_id}] {msg}"})
                      else:
                            # For less verbose mode, just add the main success/fail message
                            if "✅" in msg or "⚠️" in msg:
                               session_manager.get_current_session()["messages"].append({"role": "assistant", "content": f"[{current_session_id}] {msg}"})


                 st.rerun() # Rerun to update the main chat area


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
            st.write("This is an agentic chatbot that can process **multiple** DOCX or PDF documents (per session) and answer questions based on their combined content using function calling.") # Updated About
            st.write("It can also answer general questions and provide the current time.")
            st.write("**Note:** Direct folder upload is not supported due to browser limitations. Please select multiple files or zip a folder and upload the zip (though zip processing is not currently implemented).") # Added note about folder upload
            st.write("It uses Streamlit for the interface and OpenAI's GPT models with function calling.")
            st.write("Powered by OpenAI, Streamlit, python-docx, PyMuPDF, and Langchain/FAISS.") # Mention PyMuPDF


def render_chat():
    """Renders the main chat message area."""
    current_session = session_manager.get_current_session()
    current_messages = current_session["messages"]
    session_id = current_session.get("name", "Current Session")
    # Display list of uploaded files if multiple, or single name
    uploaded_file_names = current_session.get("uploaded_file_name", "No document uploaded")
    if isinstance(uploaded_file_names, list):
         uploaded_file_names_str = ", ".join(uploaded_file_names)
         if len(uploaded_file_names_str) > 50: # Truncate if too long
              uploaded_file_names_str = uploaded_file_names_str[:50] + "..."
         display_doc_status = f"{len(uploaded_file_names)} files processed ({uploaded_file_names_str})"
    elif uploaded_file_names:
         display_doc_status = f"'{uploaded_file_names}'"
    else:
         display_doc_status = "No document uploaded"


    # Display messages
    for message in current_messages:
        if message["role"] != "tool":
            with st.chat_message(message["role"]):
                if message.get("content") is not None:
                    if st.session_state.get("show_tool_messages") and message["role"] == "assistant" and message.get("tool_calls"):
                        if message["content"].strip() == "":
                             st.markdown(f"[{session_id}] 🤖 **Thinking... (Tool Call)**")
                        else:
                             st.markdown(message["content"])
                             st.markdown(f"[{session_id}] 🤖 **Thinking... (Tool Call)**")
                    else:
                        st.markdown(message["content"])
                elif st.session_state.get("show_tool_messages") and message["role"] == "assistant" and message.get("tool_calls"):
                     st.markdown(f"[{session_id}] 🤖 **Thinking... (Tool Call)**")


    # Chat input area
    prompt = st.chat_input(
        f"Ask about the document(s) ({display_doc_status}), current time, or general knowledge in '{session_id}'..." # Updated prompt text
    )

    return prompt
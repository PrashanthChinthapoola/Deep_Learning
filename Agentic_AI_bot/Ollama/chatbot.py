import streamlit as st
import requests
import os
import json
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from sklearn.preprocessing import normalize

import sys
import types
import torch
torch.classes = types.SimpleNamespace()
sys.modules['torch.classes'] = torch.classes

os.environ["STREAMLIT_DISABLE_FILE_WATCHER"] = "true"
# Configuration
st.set_page_config(page_title="Chatbot", page_icon="💬", layout="wide")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_chunks" not in st.session_state:
    st.session_state.file_chunks = []
if "faiss_indices" not in st.session_state:
    st.session_state.faiss_indices = []
if "chunk_texts" not in st.session_state:
    st.session_state.chunk_texts = []
if "file_sources" not in st.session_state:
    st.session_state.file_sources = []

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Ollama settings
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# ----- Utility Functions -----

def check_server():
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return r.status_code == 200
    except:
        return False

def ask_model_stream(prompt, model="llama3.2:1b"):
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True
            },
            stream=True,
            timeout=120
        )

        full_response = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                chunk = data.get("message", {}).get("content", "")
                full_response += chunk
                yield chunk
        return full_response
    except Exception as e:
        yield f"🚫 Streaming error: {str(e)}"

def extract_text(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    return ""

def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def get_relevant_chunks_faiss(query, indices, chunks, sources, top_k=4):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    query_embedding = normalize(query_embedding, axis=1)
    
    all_relevant_chunks = []
    
    # Track the sources for each retrieved chunk
    for idx, (index, chunk_set, source) in enumerate(zip(indices, chunks, sources)):
        scores, chunk_indices = index.search(query_embedding, top_k)
        
        # Add source information to each chunk
        for i in chunk_indices[0]:
            if i >= 0 and i < len(chunk_set):  # Ensure index is valid
                all_relevant_chunks.append({
                    "content": chunk_set[i],
                    "source": source,
                    "score": scores[0][chunk_indices[0].tolist().index(i)]
                })
    
    # Sort by relevance score
    all_relevant_chunks.sort(key=lambda x: x["score"], reverse=True)
    
    # Take top results overall
    return all_relevant_chunks[:top_k]

# ----- Sidebar -----

with st.sidebar:
    st.title("Chatbot")
    st.subheader("Server Status")

    if check_server():
        st.success("✅ Connected to Ollama")
    else:
        st.error("❌ Cannot connect to Ollama")
        st.info("Make sure you run `ollama serve` in terminal.")

    st.subheader("Upload Files")
    uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Reading and embedding files..."):
            # Clear previous uploads if needed
            if st.button("Clear Previous Files"):
                st.session_state.file_chunks = []
                st.session_state.faiss_indices = []
                st.session_state.chunk_texts = []
                st.session_state.file_sources = []
            
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                text = extract_text(uploaded_file)
                chunks = chunk_text(text)

                embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
                norm_embeddings = normalize(embeddings, axis=1)

                dim = norm_embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(norm_embeddings)

                st.session_state.faiss_indices.append(index)
                st.session_state.chunk_texts.append(chunks)
                st.session_state.file_sources.append(file_name)
                
            st.success(f"✅ {len(uploaded_files)} file(s) processed and indexed!")
            
            # Display files in the system
            st.subheader("Indexed Files")
            for file_name in st.session_state.file_sources:
                st.write(f"- {file_name}")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.file_chunks = []
        st.session_state.faiss_indices = []
        st.session_state.chunk_texts = []
        st.session_state.file_sources = []
        st.rerun()

# ----- Main Chat Area -----

st.title("💬 Chatbot")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input from user
prompt = st.chat_input("Ask me anything...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.faiss_indices:
        relevant_chunks = get_relevant_chunks_faiss(
            prompt, 
            st.session_state.faiss_indices, 
            st.session_state.chunk_texts,
            st.session_state.file_sources
        )
        
        # Format context with clear source attribution
        context_parts = []
        for chunk in relevant_chunks:
            context_parts.append(f"[From {chunk['source']}]:\n{chunk['content']}")
        
        context = "\n\n" + "\n\n".join(context_parts)
        
        full_prompt = (
            f"You are a helpful assistant. Use the file context if needed. "
            f"IMPORTANT: When referring to information, always mention which file it came from. "
            f"The information about different people may be in different files, so be very careful "
            f"not to mix up information between people or files.\n\n"
            f"File Context:{context}\n\n"
            f"User Question: {prompt}"
        )
    else:
        full_prompt = f"You are a helpful assistant. Answer this:\n\n{prompt}"

    with st.chat_message("assistant"):
        response_container = st.empty()
        streamed = ""
        for chunk in ask_model_stream(full_prompt):
            streamed += chunk
            response_container.markdown(streamed)

    st.session_state.messages.append({"role": "assistant", "content": streamed})

# Welcome message
if not st.session_state.messages:
    st.info("👋 Hello! Upload a file or ask me anything!")
# Import required libraries
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
import pytz
import re
import os
from io import StringIO

# For FAISS and Embedding
from sentence_transformers import SentenceTransformer
import faiss
import docx2txt
import PyPDF2
import sys
import types
import torch
torch.classes = types.SimpleNamespace()
sys.modules['torch.classes'] = torch.classes

os.environ["STREAMLIT_DISABLE_FILE_WATCHER"] = "true"
# Set up Streamlit page
st.set_page_config(page_title="Gemma Chatbot", layout="wide")
st.title("💬 Conversational Chatbot with Gemma + Document QA")
st.write("Chat with me! I can also answer from your uploaded documents.")

# Display current local time
local_time = datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
st.markdown(f"🕒 **Your Local Time:** {local_time}")

# Initialize chat history and FAISS state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your Gemma-powered chatbot. Upload a file or ask me anything!"}
    ]
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
    st.session_state.doc_chunks = []

# Sidebar for file upload
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

# Process file and update FAISS index
def process_file(file):
    text = ""
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    elif file.name.endswith(".docx"):
        text = docx2txt.process(file)
    elif file.name.endswith(".txt"):
        stringio = StringIO(file.getvalue().decode("utf-8"))
        text = stringio.read()

    # Split into chunks
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    # Embed and store in FAISS
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    st.session_state.faiss_index = index
    st.session_state.doc_chunks = chunks

if uploaded_file:
    process_file(uploaded_file)
    st.sidebar.success("✅ File processed and indexed!")

# Load LLM
llm = OllamaLLM(model="gemma3:latest")
output_parser = StrOutputParser()

# Time query helper
def get_time_response(query):
    country_timezones = {
        "india": "Asia/Kolkata", "usa": "America/New_York", "united states": "America/New_York",
        "uk": "Europe/London", "united kingdom": "Europe/London", "germany": "Europe/Berlin",
        "japan": "Asia/Tokyo", "china": "Asia/Shanghai", "australia": "Australia/Sydney",
        "canada": "America/Toronto", "brazil": "America/Sao_Paulo", "france": "Europe/Paris",
        "russia": "Europe/Moscow", "uae": "Asia/Dubai", "south africa": "Africa/Johannesburg",
        "singapore": "Asia/Singapore"
    }
    query_lower = query.lower()
    if re.search(r"\b(time|date|now|current time|what time)\b", query_lower):
        for country, timezone in country_timezones.items():
            if country in query_lower:
                now = datetime.now(pytz.timezone(timezone))
                return f"The current time in {country.title()} is {now.strftime('%A, %B %d, %Y %I:%M %p')}."
        local_now = datetime.now()
        return f"The current local time is {local_now.strftime('%A, %B %d, %Y %I:%M %p')}."
    return None

# Show chat history
for msg in st.session_state.messages:
    role = msg["role"]
    avatar = "🧑‍💻" if role == "user" else "🤖"
    st.chat_message(role, avatar=avatar).write(msg["content"])

# Handle user input
if user_input := st.chat_input("Type your message here"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user", avatar="🧑‍💻").write(user_input)

    # Check time response
    time_response = get_time_response(user_input)
    if time_response:
        st.chat_message("assistant", avatar="🤖").write(time_response)
        st.session_state.messages.append({"role": "assistant", "content": time_response})
    else:
        # Use FAISS if index exists
        context = ""
        if st.session_state.faiss_index:
            embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            query_emb = embed_model.encode([user_input])
            D, I = st.session_state.faiss_index.search(query_emb, k=3)
            context_chunks = [st.session_state.doc_chunks[i] for i in I[0]]
            context = "\n\n".join(context_chunks)

        # Prepare chat prompt
        history = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            else:
                history.append(AIMessage(content=msg["content"]))

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a helpful assistant powered by Gemma. Use the following context if relevant:\n\n{context}"),
            *history
        ])
        chain = prompt | llm | output_parser

        # Stream response
        st.session_state["full_message"] = ""
        with st.chat_message("assistant", avatar="🤖"):
            response_container = st.empty()
            for chunk in chain.stream({}):
                st.session_state["full_message"] += chunk
                response_container.write(st.session_state["full_message"])
            st.session_state.messages.append({"role": "assistant", "content": st.session_state["full_message"]})

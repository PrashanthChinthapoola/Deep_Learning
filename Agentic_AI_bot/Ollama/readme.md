# 💬 Local File-Aware Chatbot (Streamlit + Ollama + FAISS)

This project is a local chatbot app built with **Streamlit**, using **Ollama** for LLM-powered responses and **FAISS** + **Sentence-Transformers** for semantic search over uploaded files. It supports **PDF**, **DOCX**, and **TXT** files and ensures answers are accurately sourced from the uploaded documents.

---

## 🚀 Features

- 💬 Chat with your own documents
- 📄 Upload multiple PDF, DOCX, or TXT files
- 🧠 Uses `sentence-transformers` to embed document chunks
- 🔍 Retrieves relevant chunks with FAISS
- 🧑‍⚖️ Ensures context is sourced accurately per file
- ⚡ Streams responses from **Ollama API**
- 🧰 Built entirely with Streamlit

---

## 📦 Requirements

Install the required Python packages:

```bash
pip install streamlit requests PyPDF2 python-docx sentence-transformers faiss-cpu scikit-learn
```

You also need to install and run **Ollama** locally.

---

## ⚙️ Setup

1. **Install Ollama:**
   Follow instructions at [ollama.com](https://ollama.com/) to install and run locally.

2. **Pull a Model:**
   For example, pull the lightweight model used in this app:
   ```bash
   ollama pull llama3.2:1b
   ```

3. **Start the Ollama server:**
   ```bash
   ollama serve
   ```

4. **Run the chatbot app:**
   ```bash
   streamlit run chatbot.py
   ```

---

## 🧠 How It Works

1. **File Upload & Processing**
   - Reads text from PDF, DOCX, or TXT files
   - Chunks the text into manageable parts (default 1000 characters)
   - Embeds chunks using `sentence-transformers`

2. **Indexing**
   - Uses FAISS to index embeddings for similarity search

3. **Query Handling**
   - Retrieves top-k most relevant chunks across all files
   - Constructs a prompt with attribution to source files
   - Sends prompt to Ollama and streams response back

---

## 📁 File Structure

```
.
├── chatbot.py              # Main Streamlit app
├── README.md           # You're here
└── requirements.txt    # Optional: pip freeze > requirements.txt
```

---

## 📌 Notes

- Make sure the Ollama server is running before launching the app.
- The assistant will **always** mention the file source when referencing content from uploads.
- This ensures clean separation between different individuals or topics across files.



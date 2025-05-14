import os
from dotenv import load_dotenv
import streamlit as st
import openai
from langchain_openai import OpenAIEmbeddings

# --- Configuration and Setup ---
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def initialize_openai():
    """Initializes and returns the OpenAI client and embeddings model."""
    if not OPENAI_API_KEY:
        st.error("OpenAI API key not found in .env file.")
        st.markdown("Please create a `.env` file in the same directory as your script with `OPENAI_API_KEY='your_key_here'`.")
        st.stop()

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        # Test a simple call to check the key (optional, but good practice)
        # client.models.list()
        embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
        return client, embeddings_model
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client or embeddings model: {e}")
        st.markdown("Please double-check your OpenAI API key, your internet connection, and if your account has access to the embedding model (e.g., 'text-embedding-ada-002').")
        st.stop()

# Global objects initialized once
openai_client, embeddings_model = initialize_openai()
import streamlit as st
import google.generativeai as genai
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# Load a sentence transformer model (without Torch)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize Google Generative AI (Replace with your API key)
genai.configure(api_key="YOUR_GOOGLE_API_KEY")
model = genai.GenerativeModel("gemini-pro")

# In-memory knowledge base (Replace with a database for larger apps)
knowledge_base = [
    {"text": "Streamlit is an open-source Python library for building web apps."},
    {"text": "Google Generative AI helps with text-based tasks like summarization."},
    {"text": "Sentence Transformers are used for text embeddings."},
]

# Compute embeddings for the knowledge base
corpus_embeddings = np.array([embedder.encode(entry["text"]) for entry in knowledge_base])

# Fit a nearest neighbors model for similarity search
nn_model = NearestNeighbors(n_neighbors=1, metric="cosine")
nn_model.fit(corpus_embeddings)

def retrieve_context(user_query):
    """Find the most relevant context from the knowledge base."""
    query_embedding = embedder.encode([user_query])
    distances, indices = nn_model.kneighbors(query_embedding)
    
    # Return the closest matching knowledge entry
    closest_match = knowledge_base[indices[0][0]]["text"]
    return closest_match

def generate_response(user_input):
    """Generate a response using Google Gemini AI."""
    context = retrieve_context(user_input)
    prompt = f"Context: {context}\nUser: {user_input}\nAI:"

    response = model.generate_content(prompt)
    return response.text if response else "Sorry, I couldn't generate a response."

# Streamlit UI
st.title("Baymax - Your friendly neighbourhood AI")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", "")

if user_input:
    response = generate_response(user_input)
    
    st.session_state.chat_history.append({"role": "user", "text": user_input})
    st.session_state.chat_history.append({"role": "chatbot", "text": response})

    for chat in st.session_state.chat_history:
        role = "🤖" if chat["role"] == "chatbot" else "🧑"
        st.text(f"{role} {chat['text']}")

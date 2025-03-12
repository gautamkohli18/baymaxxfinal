import streamlit as st
from streamlit_lottie import st_lottie
import google.generativeai as genai
import os
import requests
import time
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide", page_title="Baymax - Friendly AI", page_icon="🤖")

def load_lottie_url(url: str):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

api_key = 'AIzaSyB3n1FTI2oiL_G7M7WqzdroNcQ-dJiFgyA'
os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=api_key)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

model_embedder = SentenceTransformer("all-MiniLM-L6-v2")
chat_memory = []

def save_memory():
    with open("chat_memory.pkl", "wb") as f:
        pickle.dump(chat_memory, f)

def load_memory():
    global chat_memory
    try:
        with open("chat_memory.pkl", "rb") as f:
            chat_memory = pickle.load(f)
    except FileNotFoundError:
        pass

load_memory()

st.sidebar.title("Chat History")
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "selected_session" not in st.session_state:
    st.session_state.selected_session = None

selected_session = st.sidebar.radio("Select a session:", list(st.session_state.chat_sessions.keys()), index=0 if st.session_state.chat_sessions else None)
if st.sidebar.button("New Chat"):
    session_id = f"Session {len(st.session_state.chat_sessions) + 1}"
    st.session_state.chat_sessions[session_id] = []
    st.session_state.selected_session = session_id
    selected_session = session_id

if selected_session:
    st.session_state.selected_session = selected_session
    chat_history = st.session_state.chat_sessions[selected_session]
else:
    chat_history = []

def typewrite_effect(text):
    placeholder = st.empty()
    typewritten_text = ""
    for char in text:
        typewritten_text += char
        placeholder.markdown(f'<div class="ai-message">{typewritten_text}</div>', unsafe_allow_html=True)
        time.sleep(0.006)
    placeholder.markdown(f'<div class="ai-message">{text}</div>', unsafe_allow_html=True)

def retrieve_context(query):
    if not chat_memory:
        return ""
    query_embedding = model_embedder.encode([query])[0].reshape(1, -1)
    similarities = [cosine_similarity(query_embedding, emb.reshape(1, -1))[0][0] for _, emb in chat_memory]
    top_indices = np.argsort(similarities)[-3:][::-1]
    return "\n".join([chat_memory[i][0] for i in top_indices])

def handle_input():
    user_input = st.session_state.user_input
    if user_input:
        context = retrieve_context(user_input)
        full_prompt = f"Context:\n{context}\n\nUser: {user_input}\nAI:"
        response = model.generate_text(full_prompt)
        chat_history.append({"role": "user", "text": user_input})
        chat_history.append({"role": "chatbot", "text": response.text})
        embedding = model_embedder.encode([user_input + response.text])[0]
        chat_memory.append((user_input + response.text, embedding))
        save_memory()
        st.session_state.user_input = ""

for message in chat_history:
    if message["role"] == "user":
        st.markdown(f'<div class="message-box"><div class="user-message">{message["text"]}</div></div>', unsafe_allow_html=True)
    else:
        typewrite_effect(message["text"])

st.text_input("You:", key="user_input", placeholder="Type your message here...", on_change=handle_input)

if st.button('Reset Chat'):
    if selected_session:
        st.session_state.chat_sessions[selected_session] = []
    st.session_state.user_input = ""

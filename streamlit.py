import nltk
import numpy as np
import json
import pickle
import random
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.models import load_model
import tensorflow as tf
import streamlit as st
from streamlit_chat import message
from datetime import datetime

# Download required NLTK data files
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents JSON file
with open('crops.json') as json_file:
    intents = json.load(json_file)

# Load pre-trained model and supporting files
words = pickle.load(open('backend/words.pkl', 'rb'))
classes = pickle.load(open('backend/classes.pkl', 'rb'))
model = load_model('backend/chatbotmodel.h5')

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Streamlit App Configuration
st.set_page_config(page_title="Smart Agriculture AI", page_icon="🌱", layout="wide")

# Custom CSS for UI Styling
st.markdown("""
    <style>
        body {
            background-color: white;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px 24px;
            border-radius: 10px;
        }
        .stTextInput>div>div>input {
            font-size: 18px;
            padding: 10px;
        }
        .chat-container {
            max-width: 800px;
            margin: auto;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.image("african_farmer.jpg", width=500)
st.title("🌱 Smart Agriculture AI Chatbot")
st.markdown("A chatbot to assist with agriculture-related inquiries using AI-powered insights.")


# About Section
st.subheader("📖 About Smart Agriculture AI")
st.markdown("""
Smart Agriculture AI is an innovative solution designed to support smallholder farmers with real-time, AI-driven insights. Our goal is to empower farmers with knowledge and tools to improve productivity and sustainability.
""")

# Features Section
st.subheader("🌟 Features")
st.markdown("""
- AI-powered chatbot for instant agricultural advice
- Supports multiple languages for accessibility
- Data-driven insights for better crop management
- Integration with Smartel's hydroponic technology
- Easy-to-use interface for farmers
""")

# Chatbot Interface
st.subheader("💬 Chat with the AI")
chat_container = st.container()
user_input = st.text_input("Type your message here:", "", key="input")

if st.button("Send"):
    if user_input:
        # Chatbot response logic here (Placeholder Response)
        response = f"🤖 AI: This is a placeholder response for '{user_input}'."
        chat_container.write(f"👤 You: {user_input}")
        chat_container.write(response)
    else:
        st.warning("Please enter a message to continue.")

# Footer Section
st.markdown("---")
st.caption("Smartel - Empowering Farmers with Climate-Resilient Solutions 🌍")

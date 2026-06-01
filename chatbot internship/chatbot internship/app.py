import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Settings
st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 FAQ Chatbot using NLP")
st.write("Ask any question related to the FAQ dataset.")

# Load FAQ dataset
faq = pd.read_csv("faq.csv")

# Convert questions into vectors
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")

faq_vectors = vectorizer.fit_transform(faq["Question"])

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_question = st.chat_input("Type your question here...")

if user_question:

    # Display user message
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.write(user_question)

    # Convert user question into vector
    user_vector = vectorizer.transform([user_question])

    # Calculate similarity
    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )

    best_match_index = similarity_scores.argmax()

    best_score = similarity_scores[0][best_match_index]

    # Generate response
    if best_score > 0.2:
        response = faq.iloc[best_match_index]["Answer"]
    else:
        response = "Sorry, I couldn't find a relevant answer."

    # Display bot response
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    with st.chat_message("assistant"):
        st.write(response)
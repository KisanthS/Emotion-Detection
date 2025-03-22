import streamlit as st
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import requests
import base64
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from dotenv import load_dotenv

import nltk
nltk.download('stopwords')  # Download stopwords dataset
from nltk.corpus import stopwords

# Load secrets from Streamlit's secrets manager
CLIENT_ID = st.secrets["SPOTIFY_CLIENT_ID"]
CLIENT_SECRET = st.secrets["SPOTIFY_CLIENT_SECRET"]

# Load environment variables from .env file
load_dotenv()
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# Load saved models
encoder = pickle.load(open('encoder.pkl', 'rb'))
tfidf = pickle.load(open('TfidfVectorizer.pkl', 'rb'))
model = tf.keras.models.load_model('my_model.h5')

# Define emojis for emotions
emoji_dict = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲",
    "love": "❤️"
}

# Text suggestions for emotions
suggestions = {
    "joy": "You're feeling great! Keep the positivity going.",
    "sadness": "It's okay to feel sad. Maybe try talking to someone you trust.",
    "anger": "Take a deep breath and try some relaxation techniques.",
    "fear": "It's normal to feel afraid sometimes. Stay grounded.",
    "surprise": "Wow, what a twist! Embrace the moment.",
    "love": "Love is beautiful. Cherish and spread kindness."
}

# Preprocessing function
ps = PorterStemmer()
def preprocess(line):
    review = re.sub('[^a-zA-Z]', ' ', line).lower().split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    return " ".join(review)

# Function to load and save emotion corrections
def load_corrections():
    try:
        with open('emotion_corrections.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_correction(user_text, corrected_emotion):
    corrections = load_corrections()
    corrections[user_text] = corrected_emotion
    with open('emotion_corrections.json', 'w') as f:
        json.dump(corrections, f)

# Emotion prediction function
def correct_emotion(user_text):
    processed_text = preprocess(user_text)
    input_array = tfidf.transform([processed_text]).toarray()
    
    corrections = load_corrections()
    if processed_text in corrections:
        predicted_emotion = corrections[processed_text]
    else:
        pred_probs = model.predict(input_array)[0]
        pred_index = np.argmax(pred_probs)
        predicted_emotion = encoder.inverse_transform([pred_index])[0]
        
    return predicted_emotion

# Function to get Spotify access token
def get_access_token():
    url = 'https://accounts.spotify.com/api/token'
    headers = {
        'Authorization': 'Basic ' + base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    }
    data = {'grant_type': 'client_credentials'}
    response = requests.post(url, headers=headers, data=data)
    return response.json().get('access_token')

# Function to fetch Tamil songs based on emotion
def search_spotify_track(emotion):
    access_token = get_access_token()
    query = f"{emotion} tamil"  # Adding "tamil" to filter Tamil songs
    url = f'https://api.spotify.com/v1/search?q={query}&type=track&limit=3'
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(url, headers=headers)
    tracks = response.json().get('tracks', {}).get('items', [])

    return [
        {
            "name": t['name'],
            "artist": t['artists'][0]['name'],
            "url": t['external_urls']['spotify'],
            "id": t['id']
        }
        for t in tracks
    ]

# -------------------- Streamlit UI --------------------

# Set Streamlit page layout
st.set_page_config(page_title="Emotion Detector 🎭", layout="centered")

# Apply Background Image
background_image_url = "https://wallpapers.com/images/featured/dark-5u7v1sbwoi6hdzsb.jpg"
st.markdown(
    f"""
    <style>
        body {{
            background: url("{background_image_url}") no-repeat center center fixed;
            background-size: cover;
        }}
        .stApp {{
            background-color: rgba(0, 0, 0, 0.5);  /* Adds a dark overlay for readability */
            padding: 20px;
            border-radius: 10px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Page Header
st.markdown("<h2 style='text-align: center; color: #ff4b4b;'>🎭 Emotion Detection App 🎭</h2>", unsafe_allow_html=True)

# User Input
user_text = st.text_input("📝 Enter your sentence:", max_chars=100)

if user_text:
    predicted_emotion = correct_emotion(user_text)

    # Display Predicted Emotion
    st.markdown(
        f"<h3 style='text-align: center;'>Predicted Emotion: {predicted_emotion.capitalize()} {emoji_dict[predicted_emotion]}</h3>",
        unsafe_allow_html=True
    )

    # Suggested Text
    st.markdown(f"<p style='text-align: center;'>{suggestions[predicted_emotion]}</p>", unsafe_allow_html=True)

    # ----------------------- 🎵 Tamil Songs Section -----------------------

    st.markdown("<hr>", unsafe_allow_html=True)  # Separator
    st.markdown("<h3 style='text-align: center; color: #1DB954;'>🎵 Suggested Songs 🎵</h3>", unsafe_allow_html=True)

    music_suggestions = search_spotify_track(predicted_emotion)
    for track in music_suggestions:
        st.markdown(
            f"<p style='text-align: center;'><strong>{track['name']}</strong> by {track['artist']}</p>",
            unsafe_allow_html=True
        )
        
        # Embed Spotify Player
        embed_url = f"https://open.spotify.com/embed/track/{track['id']}?utm_source=generator"
        st.markdown(
            f'<iframe src="{embed_url}" width="100%" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>',
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)  # Separator

    # ----------------------- 📊 Probability Graph Section -----------------------

    st.markdown("<h3 style='text-align: center; color: #FF6347;'>📊 Emotion Probability Graph 📊</h3>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(6, 3))
    pred_probs = model.predict(tfidf.transform([preprocess(user_text)]).toarray())[0]
    emotions = encoder.classes_

    sns.barplot(x=pred_probs, y=emotions, ax=ax, palette="coolwarm", edgecolor="black", linewidth=2)
    ax.set_xlabel("Probability")
    ax.set_title("Emotion Probabilities")

    st.pyplot(fig)

    st.markdown("<hr>", unsafe_allow_html=True)  # Separator

    # ----------------------- Emotion Correction -----------------------

    corrected_emotion = st.selectbox(
        "Choose the correct emotion:", list(emoji_dict.keys()), index=list(emoji_dict.keys()).index(predicted_emotion)
    )

    if corrected_emotion != predicted_emotion:
        save_correction(preprocess(user_text), corrected_emotion)
        st.success(f"Emotion corrected to: {corrected_emotion.capitalize()}!")

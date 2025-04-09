import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import re

nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# Dictionnaire des contractions anglaises
contractions = {
    "aren't": "are not", "can't": "cannot", "couldn't": "could not", 
    "didn't": "did not", "doesn't": "does not", "don't": "do not",
    "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "isn't": "is not", "shouldn't": "should not", "wasn't": "was not",
    "weren't": "were not", "won't": "will not", "wouldn't": "would not",
    "it's": "it is", "i'm": "i am", "they're": "they are", "we're": "we are",
    "you're": "you are", "i've": "i have", "you've": "you have"
}

# Liste de mots inutiles (dates, numéros, informations transactionnelles)
custom_stopwords = set([
    "date","ago", "yesterday", "today", "hour", "week", "month", "year",
    "feb","dec","january", "february","day"
    "march", "april", "may", "june","july", "august", "september", "october", "november", 
    "december","jan","day","mar","apr","aug","sep","oct","nov"
])

def preprocess_text(text):
    """Nettoyage avancé des avis Trustpilot"""

    if pd.isnull(text):
        return ""

    # Convertir en minuscule
    text = text.lower()

    # Supprimer les URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Supprimer les émojis et caractères spéciaux
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Supprime les caractères non ASCII

    # Remplacer les contractions anglaises
    for contraction, full_form in contractions.items():
        text = text.replace(contraction, full_form)

    # Supprimer les nombres
    text = re.sub(r'\d+', '', text)

    # Supprimer les expressions de date et mots transactionnels
    text = re.sub(r'\b(?:' + '|'.join(custom_stopwords) + r')\b', '', text)

    # Supprimer la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenisation et suppression des mots courts (< 3 lettres) + stopwords
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]

    return " ".join(tokens)

def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt', max_length=512, truncation=True)
    result = model(tokens)
    return int(torch.argmax(result.logits)) + 1


def scrape_reviews(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        reviews = []
        for tag in ['section']:
            for element in soup.find_all(tag, class_=re.compile(".*review.*")):
                text = element.get_text(strip=True)
                if len(text) > 50:
                    reviews.append(text)
        return list(set(reviews)) if reviews else None
    except requests.exceptions.RequestException:
        return None


def classify_sentiment(score):
    if score <= 2:
        return 'Négatif'
    elif score == 3:
        return 'Neutre'
    else:
        return 'Positif'


def analyze_reviews(df):
    df['sentiment_score'] = df['review'].apply(sentiment_score)
    df['sentiment_class'] = df['sentiment_score'].apply(classify_sentiment)
    return df


def evaluate_product(df):
    sentiment_counts = df['sentiment_class'].value_counts(normalize=True)
    if sentiment_counts.get('Positif', 0) > 0.6:
        return "✅ Ces produits sont de **BONNE qualité**. 🎉"
    else:
        return "❌ Ces produits sont de **MAUVAISE qualité**. 😞"

def nike_interface():
    st.title("👟 Analyse des Avis sur les Produits Nike")
    st.write("Cette application analyse les avis des produits Nike pour évaluer leur qualité.")

    url = st.text_input("🌐 Entrez l'URL d'une page d'avis produits :", "")

    if st.button("Analyser"):
        if url:
            with st.spinner("🔍 Analyse en cours..."):
                reviews = scrape_reviews(url)
                if reviews:
                    df = pd.DataFrame([preprocess_text(review) for review in reviews], columns=['review'])
                    df = analyze_reviews(df)
                    result = evaluate_product(df)

                    st.success(result)
                    st.write("### 📊 Résultats détaillés :")
                    st.dataframe(df[['review', 'sentiment_score', 'sentiment_class']])

                    st.write("### 📈 Répartition des Sentiments :")
                    sentiment_counts = df['sentiment_class'].value_counts()
                    st.bar_chart(sentiment_counts)
                    st.write("### 📊 Détails des proportions des sentiments")
                    st.write(sentiment_counts)
                else:
                    st.error("❗ Aucune critique trouvée ou accès bloqué.")
        else:
            st.warning("⚠️ Veuillez entrer une URL valide.")

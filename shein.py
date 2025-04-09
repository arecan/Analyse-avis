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
import re

# Téléchargement des ressources NLP
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Chargement du modèle NLP
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def preprocess_text(text):
    """Nettoyage des avis"""
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


def sentiment_score(review):
    """Analyse des sentiments"""
    tokens = tokenizer.encode(review, return_tensors='pt', max_length=512, truncation=True)
    result = model(tokens)
    return int(torch.argmax(result.logits)) + 1


def scrape_reviews(url):
    """Scraping des avis"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        reviews = []
        for tag in ['section']:
            for element in soup.find_all(tag, class_=re.compile(".*styles_reviewContentwrapper__W9Vqf.*")):
                text = element.get_text(strip=True)
                if len(text) > 50:
                    reviews.append(text)
        return list(set(reviews)) if reviews else None
    except requests.exceptions.RequestException:
        return None


def classify_sentiment(score):
    """Classification des sentiments"""
    if score <= 2:
        return 'Négatif'
    elif score == 3:
        return 'Neutre'
    else:
        return 'Positif'


def analyze_reviews(df):
    """Analyse complète des avis"""
    df['sentiment_score'] = df['review'].apply(sentiment_score)
    df['sentiment_class'] = df['sentiment_score'].apply(classify_sentiment)
    return df


def evaluate_product(df):
    """Évaluation de la qualité du produit"""
    sentiment_counts = df['sentiment_class'].value_counts(normalize=True)
    if sentiment_counts.get('Positif', 0) > 0.6:
        return "✅ Ces produits sont de **BONNE qualité**. 🎉"
    else:
        return "❌ Ces produits sont de **MAUVAISE qualité**. 😞"


def shein_interface():
    """Interface pour Shein"""
    st.title("👗 Analyse des Avis sur les Produits Shein")
    st.write("Cette application analyse les avis des produits Shein pour évaluer leur qualité.")

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
 # Footer
    st.write("---")
    st.write("💡 Projet réalisé par MOUSSAOUI Nacera.")
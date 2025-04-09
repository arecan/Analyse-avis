import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


#model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'

# Télécharger proprement les fichiers nécessaires
#tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
#model = AutoModelForSequenceClassification.from_pretrained(model_name, force_download=True)

# Téléchargement des ressources NLP
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Modèle de sentiment
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# URLs des marques
BRAND_URLS = {
    "Adidas": "https://uk.trustpilot.com/review/www.adidas.com",
    "Nike": "https://uk.trustpilot.com/review/www.nike.com",
    "Puma": "https://uk.trustpilot.com/review/puma.com",
    "Temu": "https://uk.trustpilot.com/review/temu.com"
}


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


def scrape_trustpilot_reviews(url):
    """Scrape les avis d'une page Trustpilot"""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        reviews = []
        for tag in ['section']:
            for element in soup.find_all(tag, class_=re.compile(".*review.*")):
                text = element.get_text(strip=True)
                if len(text) > 50:
                    reviews.append(text)

        return reviews if reviews else None
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors du scraping : {e}")
        return None


def analyze_brand(url):
    """Analyse les avis d'une marque et calcule les statistiques"""
    reviews = scrape_trustpilot_reviews(url)
    if not reviews:
        return None

    # Nettoyage et analyse des sentiments
    cleaned_reviews = [preprocess_text(review) for review in reviews]
    scores = [sentiment_score(review) for review in cleaned_reviews]

    # Calcul des statistiques
    positif = sum(1 for score in scores if score >= 4) / len(scores) * 100
    neutre = sum(1 for score in scores if score == 3) / len(scores) * 100
    negatif = sum(1 for score in scores if score <= 2) / len(scores) * 100
    score_moyen = sum(scores) / len(scores)

    return {
        "Positif": round(positif, 2),
        "Neutre": round(neutre, 2),
        "Négatif": round(negatif, 2),
        "Score Moyen": round(score_moyen, 2)
    }


def generate_comparison_data():
    """Récupère et analyse les avis pour toutes les marques"""
    data = []
    for brand, url in BRAND_URLS.items():
        print(f"Analyse de {brand}...")
        stats = analyze_brand(url)
        if stats:
            data.append({
                "Marque": brand,
                "Positif": stats["Positif"],
                "Neutre": stats["Neutre"],
                "Négatif": stats["Négatif"],
                "Score Moyen": stats["Score Moyen"]
            })

    return pd.DataFrame(data)


def export_to_csv(df, filename="comparaison_marques.csv"):
    """Exporte les résultats au format CSV"""
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"Fichier enregistré : {filename}")


# Pour tester indépendamment le scraping et l'analyse
if __name__ == "__main__":
    df = generate_comparison_data()
    if not df.empty:
        print(df)
        export_to_csv(df)
    else:
        print("Aucune donnée n'a été récupérée.")

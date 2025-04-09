import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from scraper import generate_comparison_data, export_to_csv

def download_csv(df):
    output = BytesIO()
    df.to_csv(output, index=False, encoding='utf-8-sig')
    output.seek(0)
    return output


def comparison_interface():
    """Interface de comparaison des marques"""
    st.title("📊 Comparaison des Sentiments des Marques")
    st.write("Comparez la répartition des sentiments des avis pour différentes marques de produits.")

    # Bouton pour lancer l'analyse
    if st.button("🔍 Lancer la comparaison"):
        with st.spinner("Analyse en cours..."):
            df = generate_comparison_data()

            if not df.empty:
                # Affichage du DataFrame
                st.write("### 💾 Données de Comparaison")
                st.dataframe(df)

                # Téléchargement des résultats au format CSV
                csv_file = download_csv(df)
                st.download_button(label="📥 Télécharger les résultats (CSV)", data=csv_file, file_name="comparaison_marques.csv", mime="text/csv")

                # Graphique en barres
                st.write("### 📈 Répartition des Sentiments par Marque")
                df_melted = df.melt(id_vars=["Marque"], value_vars=["Positif", "Neutre", "Négatif"], var_name="Sentiment", value_name="Pourcentage")
                plt.figure(figsize=(8, 6))
                sns.barplot(data=df_melted, x="Marque", y="Pourcentage", hue="Sentiment")
                plt.title("Répartition des Sentiments par Marque")
                plt.xticks(rotation=45)
                st.pyplot(plt)

                # Graphique en ligne pour le score moyen
                st.write("### 📊 Score Moyen par Marque")
                plt.figure(figsize=(8, 6))
                sns.lineplot(data=df, x="Marque", y="Score Moyen", marker="o", linewidth=2)
                plt.title("Score Moyen par Marque")
                plt.xticks(rotation=45)
                plt.grid(True)
                st.pyplot(plt)

            else:
                st.error("❗ Aucune donnée n'a été récupérée.")

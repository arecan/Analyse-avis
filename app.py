import streamlit as st
from adidas import adidas_interface
from nike import nike_interface
from puma import puma_interface
from temu import temu_interface
#from shein import shein_interface
from comparison import comparison_interface

# ✅ `st.set_page_config(...)` doit être la toute première commande !
st.set_page_config(page_title="Analyse des Avis", page_icon="📊", layout="wide")

# Barre latérale pour le choix d'analyse
st.sidebar.title("📌 Choisissez une analyse")
option = st.sidebar.radio("Sélectionnez l'analyse :", 
                          ("Analyse des Produits Adidas", 
                           "Analyse des Produits Nike", 
                           "Analyse des Produits Puma",
                           "Analyse des Produits Temu",
                           "Comparaison entre les Marques"
                           ))

# Affichage de l'interface correspondante
if option == "Analyse des Produits Adidas":
    adidas_interface()
elif option == "Analyse des Produits Nike":
    nike_interface()
elif option == "Analyse des Produits Puma":
    puma_interface()
elif option == "Comparaison entre les Marques":
    comparison_interface()
else:
    temu_interface()
st.write("---")
st.write("💡 Projet réalisé par MOUSSAOUI Nacera.")
st.write("https://uk.trustpilot.com/")

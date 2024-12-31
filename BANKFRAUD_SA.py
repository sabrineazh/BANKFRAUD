import streamlit as st

# Exemple simple d'utilisation de Streamlit
st.title('Bank Fraud Detection')
st.write("Bienvenue sur l'application de détection de fraude bancaire")

# Ouvrir le fichier Python pour lire son contenu
with open("BANK FRAUD - SABRINE AZHARI.py", "r") as f:
    code = f.read()

# Remplacer get_ipython() dans le code
code = code.replace("get_ipython()", "# get_ipython() commenté")

exec(code)



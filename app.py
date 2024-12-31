import streamlit as st
from sklearn.preprocessing import StandardScaler


# Spécifiez le chemin absolu du fichier Python
file_path = r"C:\Users\PC\BANK FRAUD - SABRINE AZHARI.py"

# Ouvrir le fichier Python généré à partir du notebook
with open(file_path, "r") as f:
    code = f.read()

# Afficher le code Python dans l'application Streamlit
st.code(code, language='python')
code = code.replace("get_ipython()", "# get_ipython() commenté car non défini en dehors de Jupyter")


# Exécuter le code Python
exec(code)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 


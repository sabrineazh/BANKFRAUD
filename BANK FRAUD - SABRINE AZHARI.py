import streamlit as st

# Exemple simple d'utilisation de Streamlit
st.title('Bank Fraud Detection')
st.write("Bienvenue sur l'application de détection de fraude bancaire")









import pandas as pd
import numpy as np

# Afficher le code Python dans l'application Streamlit

get_ipython().system('pip install -q seaborn')

st.code(code, language='python')

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.metrics import roc_auc_score
get_ipython().system('pip install -q tensorflow')
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import pandas as pd
from sklearn.ensemble import IsolationForest


# In[105]:





# ## Importation des donnees

# In[107]:

url = 'https://raw.githubusercontent.com/sabrineazh/BANKFRAUD/main/bank_transactions_data_2.csv'

# Lire les données depuis l'URL
data = pd.read_csv(url)



# In[8]:


print(data.head())


# # Analyse exploratoire des donnees 
# ## Exploration des donnees
# ### Types de variables

# In[10]:


print(data.dtypes)


# #### Identifier les variables numériques et catégorielles
# 

# In[12]:


numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = data.select_dtypes(include=['object']).columns
print("Numerical columns:", numerical_columns)
print("Categorical columns:", categorical_columns)


# ## Pre-processing des donnees

# In[16]:


#shape de notre base de sonnees
data.shape


# In[14]:


# Detection des valeurs nulles 
data.isna().sum()


# In[115]:


#Detection des valeurs dupliquees
print(f'Duplicated values: {data.duplicated().sum()}')


# Nous remarquons que nous n'avons aucune donnee manquante ni de valeurs dupliquees ce qui montre la qualite de notre base de donnee. Il n'est donc pas necessaire de faire un pre-traitement pour les donnees dans ce cas.

# In[118]:


#Detection des valeurs uniques 
data.nunique()


# In[89]:


correlation_matrix = data[numerical_columns].corr()

# Visualiser la matrice de corrélation entre valeurs numeriques
plt.figure(figsize=(5,5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matrice de Corrélation')
plt.show()


# Ces chiffres vont nous indiquer les variables sur lesquelles on va se concentrer pour notre analyse par la suite.

# ## Description statistique des variables

# In[123]:


data.describe()


# ## Analyse de distribution des variables numeriques

# In[127]:


# Histogrammes pour visualiser la distribution des variables numériques
for col in numerical_columns:
    plt.figure(figsize=(5, 3))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution de {col}')
    plt.show()


# In[128]:


#Boxplot pour les variables numeriques
for col in numerical_columns:
    sns.boxplot(x=data[col])
    plt.title(f'Bloxpot {col}')
    plt.show()


# Ici nous remarquons le distribution des donnees sur l'ensemble de notre base de donnees.

# In[130]:


#Nombres de transactions par jour
data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])
data['DayOfWeek'] = data['TransactionDate'].dt.day_name()

plt.figure(figsize=(7,4))
sns.countplot(data=data, x='DayOfWeek', order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Nombre de transactions par jour de la semaine')
plt.xlabel('Jours de la semaine')
plt.ylabel('Nombres de transactions')
plt.show()


# In[131]:


#plage horaire de transactions
data['Hour'] = data['TransactionDate'].dt.hour
plt.figure(figsize=(5,3))
sns.histplot(data=data, x='Hour',kde=True) 
plt.title('Frequence de transaction par horaire')
plt.xlabel('Heure du jour')
plt.ylabel('Frequence de transaction')
plt.show()


# In[132]:


sns.pairplot(data)


# ## Analyse de distribution des variables categorielles

# In[135]:


sns.countplot(x='Channel', data=data, hue='Channel', palette='pastel')
plt.title('Transactions par Canal')
plt.show()


# In[136]:


plt.figure(figsize=(5,5))
sns.countplot(data=data, x='Channel', hue='TransactionType', palette='viridis')
plt.xlabel('Channel')
plt.ylabel('Count')
plt.title('Nombre de transactions par canal par type')
plt.show()


# In[137]:


sns.countplot(x='CustomerOccupation', data=data)
plt.title('Nombre de transactions par metier')
plt.show()


# In[138]:


sns.countplot(x='TransactionType', data=data)
plt.title('Distribution par type de transaction')
plt.show()


# In[140]:


plt.figure(figsize=(10,6))
sns.histplot(data=data, x='CustomerAge', hue='TransactionType', multiple='stack', palette='viridis', kde=True)
plt.title('Distribution age client par Transaction Type')
plt.xlabel('Customer Age')
plt.ylabel('Frequency')
plt.show()


# In[141]:


# Répartition des montants par profession
plt.figure(figsize=(7,5))
sns.boxplot(x='CustomerOccupation', y='TransactionAmount', data=data)
plt.title("Répartition des montants de transaction par profession")
plt.xticks(rotation=45)
plt.show()


# In[142]:


# Répartition des montants de transaction par profession et par canal
plt.figure(figsize=(7, 5))
sns.boxplot(x='CustomerOccupation', y='TransactionAmount', data=data, hue='Channel',palette='pastel')
plt.title("Répartition des montants de transaction par profession et par canal")
plt.xticks(rotation=45)
plt.show()


# In[144]:


# Boxplot pour le montant de transaction par type de transaction
sns.boxplot(x='TransactionType', y='TransactionAmount', data=data)
plt.title("Montants par type de transaction")
plt.show()


# ## Statitiques descriptives pour chaque compte bancaire
# 
# ### Dans cette partie, nous allons analyser les comportements de chaque compte bancaire 

# In[146]:


device = data.groupby('AccountID')['DeviceID'].nunique()
ip_variation = data.groupby('AccountID')['IP Address'].nunique()
locations = data.groupby('AccountID')['Location'].nunique()
attempts = data.groupby('AccountID')['LoginAttempts'].nunique()

# 1. Créer une figure avec une grille de 2 lignes et 3 colonnes (pour 6 variables)
plt.figure(figsize=(15, 10))

# Graphique pour 'device' (nombre d'appareils utilisés par chaque compte)
plt.subplot(2, 3, 1)
sns.histplot(device, bins=20, color='violet', kde=True)
plt.title("Nombre d'appareils utilisés pour chaque compte")
plt.xlabel("Appareils")
plt.ylabel("Nombre de comptes")

# Graphique pour 'ip_variation' (nombre d'IP différentes utilisées par chaque compte)
plt.subplot(2, 3, 2)
sns.histplot(ip_variation, bins=20, color='purple', kde=True)
plt.title("Nombre d'IP utilisées pour chaque compte")
plt.xlabel("IP")
plt.ylabel("Nombre de comptes")

# Graphique pour 'locations' (nombre d'emplacements uniques par compte)
plt.subplot(2, 3, 4)
sns.histplot(locations, bins=20, color='blue', kde=True)
plt.title("Nombre d'emplacements uniques pour chaque compte")
plt.xlabel("Emplacements")
plt.ylabel("Nombre de comptes")

# Graphique pour 'attempts' (nombre d'essais de connexion par compte)
plt.subplot(2, 3, 5)
sns.histplot(attempts, bins=20, color='red', kde=True)
plt.title("Nombre d'essais de connexion par compte")
plt.xlabel("Essais")
plt.ylabel("Nombre de comptes")

# Ajuster l'espacement entre les graphiques pour éviter le chevauchement des titres
plt.tight_layout()

# Afficher tous les graphiques
plt.show()


# In[148]:


result = pd.DataFrame({
    'device': device,
    'ip_variation': ip_variation,
    'locations': locations,
    'attempts': attempts
})

# Afficher le DataFrame résultant
print(result)


# In[149]:


# Obtenir les valeurs maximales et minimales de chaque colonne
max_values = result.max()
min_values = result.min()

# Afficher les résultats
print("Maximales :\n", max_values)
print("\nMinimales :\n", min_values)


# Nous remarquons que certains comptes ont utilises 12 differentes appareils et 12 differentes IP adresses ou 11 differentes localisactions 
# Il y'a aussi des transactions qui ne sont passes qu'apres 4 tentatives.
# Toutes ces variables sont donc significatives et feront objet de notre etude.

# In[151]:


data['TransactionAmount'] = pd.to_numeric(data['TransactionAmount'], errors='coerce')
data['TransactionDuration'] = pd.to_numeric(data['TransactionDuration'], errors='coerce')
data['LoginAttempts'] = pd.to_numeric(data['LoginAttempts'], errors='coerce')


# In[152]:


# Calcul des statistiques par compte 
account_stats = data.groupby('AccountID').agg(
    min_transaction_amount=('TransactionAmount', 'min'),  # Montant minimum de transaction
    max_transaction_amount=('TransactionAmount', 'max'),  # Montant maximum de transaction
    min_transaction_duration=('TransactionDuration', 'min'),  # Durée minimale de transaction
    max_transaction_duration=('TransactionDuration', 'max'),  # Durée maximale de transaction
    total_login_attempts=('LoginAttempts', 'sum')  # Nombre total de tentatives de connexion
)

# Afficher les résultats
print(account_stats.head())


# In[153]:


# Boxplot pour les montants de transaction
plt.figure(figsize=(5, 3))
sns.boxplot(x=account_stats['min_transaction_amount'])
plt.title('Distribution des Montants Minimum des Transactions par Compte')
plt.xlabel('Montant Minimum de la Transaction')
plt.show()

plt.figure(figsize=(5, 3))
sns.boxplot(x=account_stats['max_transaction_amount'])
plt.title('Distribution des Montants Maximum des Transactions par Compte')
plt.xlabel('Montant Maximum de la Transaction')
plt.show()

# Boxplot pour les durées de transaction
plt.figure(figsize=(5, 3))
sns.boxplot(x=account_stats['min_transaction_duration'])
plt.title('Distribution des Durées Minimales des Transactions par Compte')
plt.xlabel('Durée Minimale de la Transaction')
plt.show()

plt.figure(figsize=(5, 3))
sns.boxplot(x=account_stats['max_transaction_duration'])
plt.title('Distribution des Durées Maximales des Transactions par Compte')
plt.xlabel('Durée Maximale de la Transaction')
plt.show()



# In[154]:


# Calcul de la plage des transactions (plage = max - min)
account_stats['transaction_range'] = account_stats['max_transaction_amount'] - account_stats['min_transaction_amount']

plt.figure(figsize=(12, 6))
sns.boxplot(x='AccountID', y='transaction_range', data=account_stats)
plt.title('Plage de Transactions par Compte')
plt.xlabel('ID du Compte')
plt.ylabel('Plage des Transactions')
plt.xticks(rotation=90)  
plt.tight_layout()
plt.show()



# ## Statistiques decriptives pour chaque groupe d'age

# In[156]:


data['AgeGroup'] = pd.cut(data['CustomerAge'], bins=[0, 25, 35, 50, 100], labels=['18-25', '26-35', '36-50', '51+'])
grouped_data = data.groupby('AgeGroup', observed=False)['TransactionAmount'].describe()
print(grouped_data)


# In[157]:


# Graphique de la moyenne de consommation par tranche d'âge
mean_consumption = data.groupby('AgeGroup', observed=False)['TransactionAmount'].mean()
plt.figure(figsize=(10, 6))
sns.barplot(x=mean_consumption.index, y=mean_consumption.values)
plt.title('Consommation Moyenne par Groupe d\'Âge')
plt.xlabel('Groupe d\'Âge')
plt.ylabel('Consommation Moyenne')
plt.show()


# In[158]:


# Histogramme pour visualiser la distribution des montants de transaction par groupe d'âge
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='TransactionAmount', hue='AgeGroup', kde=True, bins=30)
plt.title('Distribution des Montants de Transactions par Groupe d\'Âge')
plt.xlabel('Montant de la Transaction')
plt.ylabel('Fréquence')
plt.show()


# In[159]:


# Boxplot pour voir la distribution des montants de transaction par groupe d'âge
plt.figure(figsize=(10, 6))
sns.boxplot(x='AgeGroup', y='TransactionAmount', data=data)
plt.title('Distribution des Montants de Transactions par Groupe d\'Âge')
plt.xlabel('Groupe d\'Âge')
plt.ylabel('Montant de la Transaction')
plt.show()


# ## Analyses du comportement temporel des comptes

# In[161]:


# Convertir les dates en format datetime
data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])
data['PreviousTransactionDate'] = pd.to_datetime(data['PreviousTransactionDate'])

# Calculer la différence en jours entre la transaction actuelle et la précédente
data['TimeDifference'] = (data['TransactionDate'] - data['PreviousTransactionDate']).dt.total_seconds() / (60*60*24)  # en jours

# Afficher quelques résultats
print(data[['TransactionDate', 'PreviousTransactionDate', 'TimeDifference']].head())


# In[162]:


# Créer une colonne indiquant le nombre de transactions effectuées par utilisateur (ou AccountID)
data['TransactionCount'] = data.groupby('AccountID')['TransactionID'].transform('count')

# Calculer le délai moyen entre les transactions pour chaque utilisateur
data['AvgTransactionInterval'] = data.groupby('AccountID')['TimeDifference'].transform('mean')

# Afficher les informations sur les transactions fréquentes
print(data[['AccountID', 'TransactionCount', 'AvgTransactionInterval']].head())


# In[163]:


print(data['TimeDifference'])


# Nous remarquons que cette variable est biaisee et qu'on ne l'utilisera pas pour la suite de notre etude.

# ####  créer une nouvelle feature qui compare le temps de transaction avec la location afin de vérifier si deux transactions successives ont été effectuées à deux endroits différents. Pour cela, l'idée est de :
# Trier les transactions pour chaque AccountID en fonction du TransactionDate.
# Comparer la Location de chaque transaction avec celle de la transaction précédente pour le même AccountID.
# Créer une nouvelle colonne qui indique si les transactions successives ont eu lieu à des emplacements différents.
# Si elles sont différentes, la valeur sera True, sinon False.

# In[189]:


import pandas as pd

data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])

# Trier les données par AccountID et TransactionTime
data = data.sort_values(by=['AccountID', 'TransactionDate'])

# Créer une nouvelle colonne pour comparer la location entre deux transactions successives
data['LocationChange'] = data.groupby('AccountID')['Location'].shift(-1) != data['Location']

# La colonne 'LocationChange' sera True si la location de la transaction actuelle est différente de la suivante
# False sinon. Si c'est la dernière transaction d'un AccountID, cela sera NaN.

# Optionnel : Remplacer les NaN par False pour les dernières transactions
data['LocationChange'] = data['LocationChange'].fillna(False)

# Vous pouvez également créer une colonne binaire (1 ou 0) au lieu de True/False si nécessaire
data['LocationChangeBinary'] = data['LocationChange'].astype(int)

# Créer une nouvelle colonne pour la différence de temps entre la transaction actuelle et la précédente
data['TimeDifference'] = data.groupby('AccountID')['TransactionDate'].shift(-1) - data['TransactionDate']

# La colonne 'TimeDifference' contient la différence de temps entre la transaction actuelle et la suivante
# Si c'est la dernière transaction d'un AccountID, cela sera NaN.

# Optionnel : Remplacer les NaN par une valeur spécifique, par exemple '0' ou une autre valeur.
data['TimeDifference'] = data['TimeDifference'].fillna(pd.Timedelta(0))

# Affichage du résultat
print(data[['AccountID', 'TransactionDate', 'Location', 'TimeDifference']])



# In[191]:


# Filtrer les lignes où 'LocationChange' est True
transactions_with_location_change = data[data['LocationChange'] == True]

# Afficher les résultats
print(transactions_with_location_change[['AccountID', 'TransactionDate', 'Location', 'LocationChange']])


# In[193]:


data['TransactionDate'] = pd.to_datetime(data['TransactionDate']).dt.date

# Trier les données par AccountID et TransactionDate
data = data.sort_values(by=['AccountID', 'TransactionDate'])

# Créer un groupby pour chaque AccountID et TransactionDate
grouped = data.groupby(['AccountID', 'TransactionDate'])

# Filtrer les groupes qui ont plus d'une transaction avec des locations différentes
transactions_same_day_diff_locations = []

for (account_id, transaction_date), group in grouped:
    # Vérifier si la même journée a plus d'une location différente
    if len(group['Location'].unique()) > 1:
        transactions_same_day_diff_locations.append(group)

# Vérifier si nous avons des groupes valides à concaténer
if transactions_same_day_diff_locations:
    # Concaténer les résultats dans un seul DataFrame
    result = pd.concat(transactions_same_day_diff_locations)
    # Afficher les résultats
    print(result[['AccountID', 'TransactionDate', 'Location']])
else:
    print("Aucune transaction ne répond aux critères (transactions le même jour dans des locations différentes).")


# # Pre-traitement des donnees

# #### Traitement des donnees manquantes

# Nous ne disposons pas de donnes manquantes.

# #### Encodage des variables catégorielles

# In[243]:


# Convertir en variables numériques
from sklearn.preprocessing import LabelEncoder

# Colonnes à exclure de la transformation
exclude_columns = ['TransactionDate', 'PreviousTransactionDate']

# Filtrer les colonnes à encoder, exclure les colonnes spécifiques
columns_to_encode = [col for col in categorical_columns if col not in exclude_columns]

# Initialiser le LabelEncoder
label_encoder = LabelEncoder()

# Convertir les colonnes sélectionnées en variables numériques
for col in columns_to_encode:
    data[col] = label_encoder.fit_transform(data[col])

# Afficher le DataFrame après la conversion
print(data)



# # CONSTRUCTION DES MODELES ML

# Apres traitement et analyse de toutes nos donnees  nous allons d'abord commencer par la definition et preparation des variables que nous avons choisi pour la construction des modeles de Machine Learning

# ## Isolation forest

# In[245]:


import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Remplir les valeurs manquantes si nécessaire
data['TransactionDuration'] = data['TransactionDuration'].fillna(0)  

# Détecter les changements de périphérique, d'IP, et d'emplacement
data['DeviceChange'] = data.groupby('AccountID')['DeviceID'].shift(-1) != data['DeviceID']
data['IPChange'] = data.groupby('AccountID')['IP Address'].shift(-1) != data['IP Address']
data['LocationChange'] = data.groupby('AccountID')['Location'].shift(-1) != data['Location']

data['DeviceChange'] = data['DeviceChange'].fillna(False)
data['IPChange'] = data['IPChange'].fillna(False)
data['LocationChange'] = data['LocationChange'].fillna(False)

# Créer des variables binaires
data['DeviceChangeBinary'] = data['DeviceChange'].astype(int)
data['IPChangeBinary'] = data['IPChange'].astype(int)
data['LocationChangeBinary'] = data['LocationChange'].astype(int)

# Calculer la variation de solde du compte entre deux transactions
data['BalanceDifference'] = data['AccountBalance'] - data.groupby('AccountID')['AccountBalance'].shift(1)
data['BalancePercentageChange'] = (data['BalanceDifference'] / data.groupby('AccountID')['AccountBalance'].shift(1)) * 100

# Sélectionner les variables pour la détection d'anomalies
features = [
    'TransactionAmount', 'TransactionDuration', 'TransactionCount', 'AvgTransactionInterval',
    'DeviceChangeBinary', 'IPChangeBinary', 'LocationChangeBinary',
    'BalancePercentageChange', 'LoginAttempts'
]

# Remplir les valeurs manquantes
X = data[features].fillna(0)

# Appliquer l'Isolation Forest pour détecter les anomalies
model = IsolationForest(contamination=0.05, random_state=42)  # 'contamination' indique le taux estimé d'anomalies
data['Anomaly_Score'] = model.fit_predict(X)  # Prédire les anomalies

# La colonne 'Anomaly_Score' contient 1 pour les points normaux, -1 pour les anomalies
data['Anomaly_Score'] = data['Anomaly_Score'].map({1: 0, -1: 1})  # 0 = normal, 1 = anomalie

# Affichage des anomalies détectées
anomalies = data[data['Anomaly_Score'] == 1]
print(f"Nombre d'anomalies détectées : {len(anomalies)}")

#  Visualiser les anomalies détectées
plt.figure(figsize=(10, 6))
plt.scatter(data.index, data['TransactionAmount'], c=data['Anomaly_Score'], cmap='coolwarm')
plt.title('Détection d\'Anomalies - Isolation Forest')
plt.xlabel('Index')
plt.ylabel('Transaction Amount')
plt.show()

# Affichage des anomalies
print(anomalies[['TransactionID', 'TransactionAmount', 'Anomaly_Score']].head())




# ## DBSCAN Density-Based Spatial Clustering of Applications with Noise
# 

# In[263]:


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Sélectionner les features
features = [  'TransactionAmount', 'TransactionDuration', 'TransactionCount', 'AvgTransactionInterval', 
    'DeviceChangeBinary', 'IPChangeBinary', 'LocationChangeBinary',
    'BalancePercentageChange', 'LoginAttempts'
]

X = data[features].fillna(0)

# Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Appliquer DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=4)
data['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# Anomalies (bruit)
data['Anomaly'] = (data['DBSCAN_Cluster'] == -1).astype(int)

print(f"Nombre d'anomalies détectées par DBSCAN : {data['Anomaly'].sum()}")

# Affichage du tableau des anomalies détectées
anomalies = data[data['Anomaly'] == 1]
print(anomalies[['TransactionID', 'TransactionAmount', 'TransactionDate', 'Anomaly']])

plt.scatter(data.index, data['TransactionAmount'], c=data['Anomaly'], cmap='coolwarm')
plt.title('Détection d\'Anomalies - DBSCAN')
plt.xlabel('Index')
plt.ylabel('Transaction Amount')
plt.show()


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['TransactionAmount'], data['TransactionDuration'], data['TransactionCount'], 
           c=data['Anomaly'], cmap='coolwarm', marker='o')
ax.set_title('Détection d\'Anomalies - DBSCAN (3D)')
ax.set_xlabel('Transaction Amount')
ax.set_ylabel('Transaction Duration')
ax.set_zlabel('Transaction Count')
plt.show()


# ## OneClassSVM

# In[265]:


from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Sélectionner les features
features = [  'TransactionAmount', 'TransactionDuration', 'TransactionCount', 'AvgTransactionInterval', 
    'DeviceChangeBinary', 'IPChangeBinary', 'LocationChangeBinary',
    'BalancePercentageChange', 'LoginAttempts']

X = data[features].fillna(0)

# Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Appliquer One-Class SVM
ocsvm = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')  
data['OneClassSVM_Prediction'] = ocsvm.fit_predict(X_scaled)

# Anomalies (valeurs -1 sont les anomalies)
data['Anomaly'] = (data['OneClassSVM_Prediction'] == -1).astype(int)

# Affichage du nombre d'anomalies détectées par One-Class SVM
print(f"Nombre d'anomalies détectées par One-Class SVM : {data['Anomaly'].sum()}")

# Affichage du tableau des anomalies détectées
anomalies = data[data['Anomaly'] == 1]
print(anomalies[['TransactionID', 'TransactionAmount', 'TransactionDate', 'Anomaly']])



# ## K-means Clustering pour la détection des anomalies
# 

# In[251]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Sélectionner les variables pour le clustering
features = [
    'TransactionAmount', 'TransactionDuration', 'TransactionCount', 'AvgTransactionInterval', 
    'DeviceChangeBinary', 'IPChangeBinary', 'LocationChangeBinary',
    'BalancePercentageChange', 'LoginAttempts'
]

# Remplir les valeurs manquantes
X = data[features].fillna(0)

# Initialiser le scaler avant de l'utiliser
scaler = StandardScaler()

# Appliquer le scaling
X_scaled = scaler.fit_transform(X)

# Méthode du coude pour choisir k optimal
inertia = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)



# Appliquer K-means clustering avec k=2
kmeans = KMeans(n_clusters=2, random_state=0)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Calculer la distance de chaque point au centre de son cluster
data['DistanceToCentroid'] = [np.linalg.norm(x - kmeans.cluster_centers_[cluster]) 
                              for x, cluster in zip(X_scaled, data['Cluster'])]

# Détecter les anomalies comme les points éloignés du centre des clusters
threshold = data['DistanceToCentroid'].quantile(0.95)  # Anomalies au-delà du 95ème percentile
data['Anomaly'] = (data['DistanceToCentroid'] > threshold).astype(int)

# Affichage des anomalies
anomalies = data[data['Anomaly'] == 1]
print(f"Nombre d'anomalies détectées : {len(anomalies)}")
print(anomalies[['TransactionID', 'TransactionAmount', 'TransactionDate', 'Anomaly']])





# ## Autoencoder

# In[787]:


# Sélectionner les variables pour l'autoencodeur
features = [
    'TransactionAmount', 'TransactionDuration', 'TransactionCount', 
    'DeviceChangeBinary', 'IPChangeBinary', 'LocationChangeBinary',
    'BalancePercentageChange', 'LoginAttempts'
]

# Remplir les valeurs manquantes
X = data[features].fillna(0)

# Normaliser les données
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Construire l'autoencodeur
autoencoder = Sequential([
    Dense(64, activation='relu', input_dim=X_scaled.shape[1]),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(X_scaled.shape[1], activation='sigmoid')
])

# Compiler et entraîner l'autoencodeur
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=256, validation_split=0.2, verbose=0)

# Reconstruire les données
reconstructed = autoencoder.predict(X_scaled)

# Calculer l'erreur de reconstruction
reconstruction_error = np.mean(np.abs(reconstructed - X_scaled), axis=1)

# Détecter les anomalies en fonction de l'erreur de reconstruction
threshold = np.percentile(reconstruction_error, 95)  # Anomalies au-dessus du 95ème percentile
data['Anomaly'] = (reconstruction_error > threshold).astype(int)

# Affichage des anomalies
anomalies = data[data['Anomaly'] == 1]
print(f"Nombre d'anomalies détectées : {len(anomalies)}")

print(anomalies[['TransactionID', 'TransactionAmount', 'Anomaly']])





# ## Local Outlier Factor

# In[795]:


from sklearn.neighbors import LocalOutlierFactor

# Appliquer LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
data['Anomaly_LOF'] = lof.fit_predict(X_scaled)

# Les résultats de LOF sont -1 pour les anomalies et 1 pour les points normaux
data['Anomaly_LOF'] = (data['Anomaly_LOF'] == -1).astype(int)

print(f"Nombre d'anomalies détectées par LOF : {data['Anomaly_LOF'].sum()}")

# Ajouter une colonne 'Anomaly' pour indiquer les anomalies détectées
data['Anomaly'] = data['Anomaly_LOF']

# Filtrer les anomalies (lignes où Anomaly est 1)
anomalies = data[data['Anomaly'] == 1]

# Imprimer les colonnes spécifiques des anomalies
print(anomalies[['TransactionID', 'TransactionAmount', 'Anomaly']])

# Assurez-vous d'avoir une colonne 'Anomaly' dans le DataFrame (qui indique si la transaction est une anomalie)
data['Anomaly'] = data['Anomaly_LOF']




# ## Gaussian Mixture

# In[427]:


from sklearn.mixture import GaussianMixture

# Appliquer GMM
gmm = GaussianMixture(n_components=5, random_state=42)
data['GMM_Cluster'] = gmm.fit_predict(X_scaled)

# Calculer les probabilités d'appartenance à chaque cluster
data['GMM_Probabilities'] = gmm.predict_proba(X_scaled).max(axis=1)

# Anomalies : probabilités trop faibles
threshold = data['GMM_Probabilities'].quantile(0.05)
data['Anomaly_GMM'] = (data['GMM_Probabilities'] < threshold).astype(int)

print(f"Nombre d'anomalies détectées par GMM : {data['Anomaly_GMM'].sum()}")




# ## Agglomerative Clustering

# In[801]:


from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Appliquer Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=5)  # Vous pouvez ajuster le nombre de clusters ici
data['Agglomerative_Cluster'] = agg_clustering.fit_predict(X_scaled)

# Calculer la distance de chaque point par rapport au centre de son cluster (en utilisant les centres des clusters)
# Les clusters ne sont pas explicitement définis, donc nous utilisons la méthode de moyenne des points
# dans chaque cluster pour approximer la distance

# Créer une colonne pour stocker la distance au centre du cluster
data['DistanceToCentroid'] = np.nan

for cluster_id in np.unique(data['Agglomerative_Cluster']):
    # Calculer le centre du cluster
    cluster_center = X_scaled[data['Agglomerative_Cluster'] == cluster_id].mean(axis=0)
    
    # Calculer la distance de chaque point à ce centre
    data.loc[data['Agglomerative_Cluster'] == cluster_id, 'DistanceToCentroid'] = \
        np.linalg.norm(X_scaled[data['Agglomerative_Cluster'] == cluster_id] - cluster_center, axis=1)

# Détecter les anomalies comme les points éloignés du centre des clusters
threshold = data['DistanceToCentroid'].quantile(0.95)  # Anomalies au-delà du 95ème percentile
data['Anomaly'] = (data['DistanceToCentroid'] > threshold).astype(int)

# Affichage du nombre d'anomalies détectées
print(f"Nombre d'anomalies détectées par Agglomerative Clustering : {data['Anomaly'].sum()}")

# Affichage des anomalies détectées
anomalies = data[data['Anomaly'] == 1]
print(anomalies[['TransactionID', 'TransactionAmount', 'TransactionDate', 'Anomaly']])




# Les résultats des anomalies détectées par Isolation Forest, Autoencoder, et LOF sont cohérents (126 anomalies détectées dans chaque cas).
# Le score de Silhouette pour K-Means est relativement bas, ce qui indique que les clusters ne sont pas bien séparés, alors que le GMM a un score de Silhouette plus élevé, indiquant une séparation meilleure entre les groupes.

# # Evaluation des modeles
# ## Visualisation PCA

# In[295]:


# Charger les données
features = [
    'TransactionAmount', 'TransactionDuration', 'TransactionCount', 'AvgTransactionInterval',
    'DeviceChangeBinary', 'IPChangeBinary', 'LocationChangeBinary',
    'BalancePercentageChange', 'LoginAttempts'
]
X = data[features].fillna(0)

# Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Utilisation de PCA pour une visualisation facile en 2D ou 3D
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# -----------------------------------------
# 1. Isolation Forest
# -----------------------------------------
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest_labels = iso_forest.fit_predict(X_scaled)
iso_forest_labels = np.where(iso_forest_labels == -1, 1, 0)  # 1 = anomalie, 0 = normal

# Visualisation des anomalies détectées
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iso_forest_labels, cmap='coolwarm')
plt.title('Anomalies détectées par Isolation Forest')
plt.show()

# -----------------------------------------
# 2. KMeans
# -----------------------------------------
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_labels = np.where(kmeans_labels == 1, 1, 0)  # Considérer le cluster avec les anomalies comme 1

# Visualisation des anomalies détectées
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='coolwarm')
plt.title('Anomalies détectées par KMeans')
plt.show()

# -----------------------------------------
# 3. One-Class SVM
# -----------------------------------------
one_class_svm = OneClassSVM(nu=0.05, kernel="rbf", gamma='auto')
one_class_svm_labels = one_class_svm.fit_predict(X_scaled)
one_class_svm_labels = np.where(one_class_svm_labels == -1, 1, 0)

# Visualisation des anomalies détectées
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=one_class_svm_labels, cmap='coolwarm')
plt.title('Anomalies détectées par One-Class SVM')
plt.show()

# -----------------------------------------
# 4. Autoencoder
# -----------------------------------------
# Construire un autoencodeur simple
input_layer = Input(shape=(X_scaled.shape[1],))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(X_scaled.shape[1], activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Entraîner l'autoencodeur
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)

# Obtenez les reconstructions et calculez les erreurs de reconstruction
reconstructed = autoencoder.predict(X_scaled)
reconstruction_error = np.mean(np.square(X_scaled - reconstructed), axis=1)

# Utiliser un seuil pour détecter les anomalies (erreur supérieure au 95e percentile)
threshold = np.percentile(reconstruction_error, 95)
autoencoder_labels = np.where(reconstruction_error > threshold, 1, 0)

# Visualisation des anomalies détectées
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=autoencoder_labels, cmap='coolwarm')
plt.title('Anomalies détectées par Autoencoder')
plt.show()

# -----------------------------------------
# 5. DBSCAN
# -----------------------------------------
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
dbscan_labels = np.where(dbscan_labels == -1, 1, 0)  # -1 indique les anomalies

# Visualisation des anomalies détectées
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='coolwarm')
plt.title('Anomalies détectées par DBSCAN')
plt.show()

# -----------------------------------------
# 6. Local Outlier Factor (LOF)
# -----------------------------------------
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof_labels = lof.fit_predict(X_scaled)
lof_labels = np.where(lof_labels == -1, 1, 0)  # -1 indique les anomalies

# Visualisation des anomalies détectées
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=lof_labels, cmap='coolwarm')
plt.title('Anomalies détectées par LOF')
plt.show()

# -----------------------------------------
# 7. Gaussian Mixture Model (GMM)
# -----------------------------------------
gmm = GaussianMixture(n_components=2, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
gmm_labels = np.where(gmm_labels == 1, 1, 0)  # Considérer le cluster avec les anomalies comme 1

# Visualisation des anomalies détectées
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='coolwarm')
plt.title('Anomalies détectées par GMM')
plt.show()




# In[281]:


# Évaluation par Silhouette Score
silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)
silhouette_dbscan = silhouette_score(X_scaled, dbscan_labels)

print(f"Silhouette Score pour KMeans: {silhouette_kmeans}")
print(f"Silhouette Score pour DBSCAN: {silhouette_dbscan}")


# # Tuning des modeles 

# In[322]:


# Tuning des hyperparamètres et entraînement du modèle
model = IsolationForest(contamination=0.1, n_estimators=200, max_samples=0.8)
model.fit(X)

# Prédiction des anomalies (-1 = anomalie, 1 = normal)
anomalies_if = model.predict(X)

# Affichage du nombre d'anomalies détectées
anomalies_detected = (anomalies_if == -1).sum()
print(f"Nombre d'anomalies détectées : {anomalies_detected}")

# Visualisation des anomalies détectées (si X a 2 dimensions, cela fonctionne bien pour un plot 2D)
plt.figure(figsize=(10, 6))

# Points normaux (anomalies_if == 1)
normal_points = X.iloc[anomalies_if == 1]  # Extraire les points normaux avec .iloc
plt.scatter(normal_points.iloc[:, 0], normal_points.iloc[:, 1], c='blue', label='Normal', s=10)

# Anomalies détectées (anomalies_if == -1)
anomalous_points = X.iloc[anomalies_if == -1]  # Extraire les anomalies avec .iloc
plt.scatter(anomalous_points.iloc[:, 0], anomalous_points.iloc[:, 1], c='red', label='Anomalie', s=20)

plt.title("Détection des anomalies avec Isolation Forest")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()


# In[314]:


from sklearn.model_selection import GridSearchCV

# Définir les paramètres à tester
param_grid = {
    'n_clusters': [2, 3, 4, 5, 6],
    'init': ['k-means++', 'random'],
    'max_iter': [100, 200, 300],
    'tol': [1e-4, 1e-3, 1e-2]
}

# Appliquer la recherche sur grille
kmeans = KMeans(random_state=42)
grid_search = GridSearchCV(kmeans, param_grid, cv=3)
grid_search.fit(X_scaled)

# Afficher les meilleurs paramètres
print("Meilleurs paramètres:", grid_search.best_params_)


# In[334]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

# Définir un modèle d'autoencodeur
def create_autoencoder(input_dim, encoding_dim=32):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')
    return autoencoder

# Utiliser une classe personnalisée pour intégrer avec GridSearchCV
class KerasAutoencoder(BaseEstimator):
    def __init__(self, encoding_dim=32, epochs=50, batch_size=32, learning_rate=0.001):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
    
    def fit(self, X, y=None):
        self.autoencoder_ = create_autoencoder(X.shape[1], self.encoding_dim)
        self.autoencoder_.fit(X, X, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self
    
    def predict(self, X):
        return self.autoencoder_.predict(X)
    
    def score(self, X, y=None):
        # Calculer l'erreur quadratique moyenne sur les données de validation
        X_pred = self.predict(X)
        mse = mean_squared_error(X, X_pred)
        return -mse  # GridSearchCV maximise donc on retourne le négatif de MSE

# Paramètres à tester pour le tuning
param_grid = {
    'encoding_dim': [16, 32, 64],
    'epochs': [50, 100],
    'batch_size': [16, 32],
    'learning_rate': [0.001, 0.01]
}

# Générer des données fictives pour l'exemple
X = np.random.rand(100, 10)  # 100 échantillons, 10 caractéristiques

# Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Utiliser GridSearch avec la classe personnalisée KerasAutoencoder
autoencoder = KerasAutoencoder()

# GridSearch avec les paramètres spécifiés
grid = GridSearchCV(estimator=autoencoder, param_grid=param_grid, n_jobs=1, cv=3)

# Ajuster le modèle
grid_result = grid.fit(X_scaled)

# Afficher les meilleurs paramètres et le meilleur score
print("Meilleurs paramètres trouvés : ", grid_result.best_params_)
print("Meilleur score de perte : ", grid_result.best_score_)

# Afficher les résultats de la recherche
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_result.cv_results_['params']):
    print(f"{mean:.4f} (+/-{std:.4f}) avec {params}")


# In[342]:


get_ipython().system('jupyter nbconvert --to python "BANK FRAUD - SABRINE AZHARI.ipynb"')


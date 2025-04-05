# models/train_model.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from models.utils_model import carregar_dados, colunas_features
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Carrega e prepara os dados
df = carregar_dados()
df = df.dropna(subset=colunas_features + ['nome_cientifico'])

# Codifica a variável alvo
le = LabelEncoder()
df['planta_codificada'] = le.fit_transform(df['nome_cientifico'])

# Separa X e y
X = df[colunas_features]
y = df['planta_codificada']

# Treina o modelo
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X, y)

# Salva o modelo e o encoder
joblib.dump(modelo, 'modelo_random_forest.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("✅ Modelo e encoder salvos com sucesso.")

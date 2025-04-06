import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import sys
import os

# Adiciona o caminho do diretório pai para importar utilitários
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.utils_model import carregar_dados, colunas_features

# Carrega e prepara os dados
df = carregar_dados()

# Remove linhas com valores NaN nas colunas de interesse
df = df.dropna(subset=colunas_features + ['nome_cientifico'])

# Codifica a variável alvo
le = LabelEncoder()
df['planta_codificada'] = le.fit_transform(df['nome_cientifico'])

# Separa features e alvo
X = df[colunas_features]
y = df['planta_codificada']

# Normaliza as features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Verificações de integridade
assert len(X_train) == len(y_train), "X_train e y_train com tamanhos diferentes!"
assert len(X_test) == len(y_test), "X_test e y_test com tamanhos diferentes!"

# Treina o modelo RandomForest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Faz predições
y_pred = rf.predict(X_test)

# Avaliação do modelo
accuracy_rf = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy_rf:.2f}")

# Garante que apenas as classes presentes no y_test sejam usadas no relatório
classes_presentes = np.unique(y_test)
nomes_presentes = le.inverse_transform(classes_presentes)
relatorio = classification_report(y_test, y_pred, labels=classes_presentes, target_names=nomes_presentes)

# Exibe e salva o relatório
print("\nRelatório de Classificação:\n")
print(relatorio)
with open('models/relatorio_classificacao.txt', 'w', encoding='utf-8') as f:
    f.write(relatorio)

# Salva o modelo, scaler e label encoder
joblib.dump(rf, 'models/random_forest_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
print("\nModelo, scaler e label encoder salvos com sucesso!")

# Gráfico de importância das features
importances = rf.feature_importances_
indices = importances.argsort()[::-1]
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.title("Importância das Features - RandomForest")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig('models/importancia_features.png')
plt.show()

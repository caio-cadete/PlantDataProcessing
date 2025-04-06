import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier  # Usando RandomForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score  # Para validação cruzada
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV  # Para ajuste de hiperparâmetros
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.utils_model import carregar_dados, colunas_features
from sklearn.preprocessing import StandardScaler  # Para normalização

# Carrega e prepara os dados
df = carregar_dados()

# Remove linhas com valores NaN
df = df.dropna(subset=colunas_features + ['nome_cientifico'])

# Codifica a variável alvo 'nome_cientifico'
le = LabelEncoder()
df['planta_codificada'] = le.fit_transform(df['nome_cientifico'])

# Separa as features (X) e a variável alvo (y)
X = df[colunas_features]
y = df['planta_codificada']

# Normaliza as variáveis numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divida os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Verificar se X_train e y_train têm o mesmo comprimento após a divisão
assert len(X_train) == len(y_train), "O número de amostras em X_train e y_train deve ser o mesmo."
assert len(X_test) == len(y_test), "O número de amostras em X_test e y_test deve ser o mesmo."

# Ajuste de Hiperparâmetros utilizando GridSearchCV para o RandomForest
param_grid = {
    'n_estimators': [50, 100, 150],  # Número de árvores na floresta
    'max_depth': [10, 20, 30, None],  # Profundidade máxima das árvores
    'min_samples_split': [2, 5, 10],  # Número mínimo de amostras para dividir um nó
    'min_samples_leaf': [1, 2, 4],    # Número mínimo de amostras em um nó folha
    'bootstrap': [True, False]        # Usar bootstrap ou não
}

# Cria o modelo de RandomForest
rf = RandomForestClassifier(random_state=42)

# Realiza uma busca em grid para otimizar os hiperparâmetros
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Melhores parâmetros encontrados
print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")

# Usa o modelo com os melhores parâmetros encontrados
modelo_rf = grid_search.best_estimator_

# Fazendo previsões no conjunto de teste
y_pred_rf = modelo_rf.predict(X_test)

# Calculando a acurácia
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Acurácia no conjunto de teste (RandomForest): {accuracy_rf:.2f}")

# Relatório detalhado de desempenho
report_rf = classification_report(y_test, y_pred_rf)
print(report_rf)

# Salva o modelo e o LabelEncoder
joblib.dump(modelo_rf, 'modelo_random_forest.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("✅ Modelo e encoder salvos com sucesso.")

# Verificando a importância das variáveis
importancia_rf = modelo_rf.feature_importances_

# Criando um gráfico para visualizar as importâncias
plt.figure(figsize=(10, 6))
plt.barh(colunas_features, importancia_rf)
plt.xlabel('Importância')
plt.title('Importância das Variáveis no Modelo Random Forest')
plt.show()

# Identificar as variáveis com baixa importância
variaveis_baixa_importancia_rf = [colunas_features[i] for i in range(len(importancia_rf)) if importancia_rf[i] < 0.01]
print("Variáveis com baixa importância (Random Forest):", variaveis_baixa_importancia_rf)

# Exibir as variáveis com

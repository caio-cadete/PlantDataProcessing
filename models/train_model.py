import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
# Importa as funções e variáveis do seu módulo
from models.utils_model import carregar_dados, colunas_features

# Pasta de saída
PASTA_MODELOS = "models"
os.makedirs(PASTA_MODELOS, exist_ok=True)

def preparar_dados(df, target_col):
    print(f"🔍 Preparando dados para '{target_col}'...")
    df = df.dropna(subset=colunas_features + [target_col])
    le = LabelEncoder()
    df['target_encoded'] = le.fit_transform(df[target_col])
    X = df[colunas_features]
    y = df['target_encoded']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled_df, y, le, scaler

from sklearn.utils.class_weight import compute_class_weight

def treinar_e_salvar_modelo(X, y, le, scaler, target_col, return_preds=False):
    print("🚀 Iniciando treino do modelo...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # # 🔹 Calcula os pesos automaticamente com base na distribuição das classes
    # classes = np.unique(y_train)
    # class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    # class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}

    # print(f"⚖️ Pesos calculados para '{target_col}':")
    # for k, v in class_weight_dict.items():
    #     print(f"  Classe {le.inverse_transform([k])[0]}: peso {v:.2f}")

    # 🔹 Treina o modelo com os pesos
    modelo = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        # class_weight=class_weight_dict
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Avaliação
    nomes_classes = le.inverse_transform(np.unique(y_test))
    relatorio = classification_report(y_test, y_pred, target_names=nomes_classes, zero_division=0)
    relatorio_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    df_relatorio = pd.DataFrame(relatorio_dict).T
    
    df_precision_zero = df_relatorio[(df_relatorio['precision'] == 0) & (df_relatorio['support'] > 0)]
    
    if not df_precision_zero.empty:
        print("\n🚨 Classes com precisão zero:")
        print(df_precision_zero)
    else:
        print("\n✅ Nenhuma classe com precisão zero.")

    # Salva modelo, encoder e scaler
    joblib.dump(modelo, f"{PASTA_MODELOS}/modelo_{target_col}.pkl")
    joblib.dump(le, f"{PASTA_MODELOS}/label_encoder_{target_col}.pkl")
    joblib.dump(scaler, f"{PASTA_MODELOS}/scaler_{target_col}.pkl")

    with open(f"{PASTA_MODELOS}/relatorio_{target_col}.txt", "w", encoding="utf-8") as f:
        f.write(relatorio)

    print(f"✅ Modelo '{target_col}' treinado com acurácia: {acc:.4f}")

    if return_preds:
        return modelo, X_test, y_test, y_pred
    return modelo


def plotar_importancia(modelo, target_col):
    importances = modelo.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 6))
    plt.title(f"Importância das Features - {target_col}")
    plt.bar(range(len(importances)), importances[indices])
    todas_features = colunas_features
    plt.xticks(range(len(importances)), [todas_features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(f"{PASTA_MODELOS}/importancia_{target_col}.png")
    plt.close()
    print(f"📈 Gráfico de importância das features salvo para '{target_col}'.")

# ========== EXECUÇÃO ==========

df = carregar_dados()

alvos = ['classe', 'ordem', 'familia', 'genero', 'nome_cientifico']

print("\n📦 Iniciando treinamento dos modelos...\n")

for target in tqdm(alvos, desc="🔁 Processando alvos"):
    print(f"\n==============================")
    print(f"🎯 Treinando modelo para: {target.upper()}")
    print(f"==============================")
    try:
        X, y, le, scaler = preparar_dados(df.copy(), target)
        modelo, X_test, y_test, y_pred = treinar_e_salvar_modelo(X, y, le, scaler, target, return_preds=True)

        print("\n📊 Relatório de Classificação:")
        print(classification_report(y_test, y_pred, target_names=le.inverse_transform(sorted(set(y_test)))))

        plotar_importancia(modelo, target)
        # 🔽 Salvando as features reais utilizadas no treino (após dropna)
        features_utilizadas = list(X.columns)
        with open(f"{PASTA_MODELOS}/features_{target}.txt", "w", encoding="utf-8") as f:
            for feat in features_utilizadas:
                f.write(f"{feat}\n")
    except Exception as e:
        print(f"⚠️ Erro ao treinar para '{target}': {e}")


import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Features usadas nos modelos
from models.utils_model import colunas_features

# Caminho da pasta com os modelos
PASTA_MODELOS = "models"

# Ordem dos alvos para predição em cascata
ALVOS = ['classe', 'ordem', 'familia', 'genero', 'nome_cientifico']


def carregar_artefatos(target):
    print(f"🔍 Carregando modelo, encoder e scaler para: {target}")
    try:
        modelo = joblib.load(os.path.join(PASTA_MODELOS, f"modelo_{target}.pkl"))
        encoder = joblib.load(os.path.join(PASTA_MODELOS, f"label_encoder_{target}.pkl"))
        scaler = joblib.load(os.path.join(PASTA_MODELOS, f"scaler_{target}.pkl"))
        return modelo, encoder, scaler
    except Exception as e:
        print(f"❌ Erro ao carregar artefatos para '{target}': {e}")
        raise


def predizer_em_cascata(dados_input):
    print("\n🚀 Iniciando predição em cascata...\n")
    
    dados = dados_input[colunas_features].copy()
    historico_preds = {}

    for alvo in ALVOS:
        print(f"\n📌 Etapa: {alvo.upper()}")

        # Carrega os artefatos do modelo
        modelo, encoder, scaler = carregar_artefatos(alvo)

        # Garante que só as features corretas sejam usadas
        dados_para_escalar = dados[colunas_features]
        dados_escalados = scaler.transform(dados_para_escalar)

        # Predição
        pred_codificada = modelo.predict(dados_escalados)
        pred_nome = encoder.inverse_transform(pred_codificada)

        # Feedback para o usuário
        for i, nome in enumerate(pred_nome):
            print(f"🔸 Amostra {i+1} → {alvo}: {nome}")

        # Armazena a predição
        historico_preds[alvo] = pred_nome

        # Adiciona como nova feature para a próxima rodada (opcional)
        dados[f"pred_{alvo}"] = pred_codificada

    print("\n✅ Predição finalizada com sucesso!")
    return pd.DataFrame(historico_preds)


# ========== TESTE COM UMA AMOSTRA ==========

if __name__ == "__main__":
    from models.utils_model import carregar_dados

    # Carrega dados e seleciona uma amostra
    df = carregar_dados()
    amostras_para_predizer = df.sample(30, random_state=42)  # <-- Pode aumentar para avaliação mais completa

    print("🧪 Rodando pipeline para amostras:\n")
    print(amostras_para_predizer[colunas_features])

    resultado = predizer_em_cascata(amostras_para_predizer)

    print("\n📋 Resultado da predição:")
    print(resultado)

    # Avaliação se houver rótulo verdadeiro
    if 'nome_cientifico' in amostras_para_predizer.columns:
        y_true = amostras_para_predizer['nome_cientifico'].values
        y_pred = resultado['nome_cientifico'].values

        print(f"\n🎯 Acurácia da predição do nome científico: {accuracy_score(y_true, y_pred):.4f}")

        # Relatório de classificação
        print("\n📈 Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))

        # Matriz de confusão (plot simples via seaborn)
        try:
            cm = confusion_matrix(y_true, y_pred, labels=np.unique(np.concatenate((y_true, y_pred))))
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=np.unique(y_true),
                        yticklabels=np.unique(y_true))
            plt.xlabel('Predito')
            plt.ylabel('Real')
            plt.title('🧩 Matriz de Confusão - Nome Científico')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"⚠️ Erro ao gerar matriz de confusão: {e}")

        # Comparativo direto
        comparativo = pd.DataFrame({
            'Real': y_true,
            'Previsto': y_pred
        })
        print("\n📌 Comparativo real x previsto:")
        print(comparativo)

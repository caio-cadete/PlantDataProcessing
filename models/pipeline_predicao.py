import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import joblib
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

        with open(os.path.join(PASTA_MODELOS, f"features_{target}.txt")) as f:
            features = f.read().splitlines()

        return modelo, encoder, scaler, features
    except Exception as e:
        print(f"❌ Erro ao carregar artefatos para '{target}': {e}")
        raise


def predizer_em_cascata(dados_input):
    print("\n🚀 Iniciando predição em cascata...\n")
    
    dados = dados_input.copy().reset_index(drop=True)
    historico_preds = {}

    for alvo in ALVOS:
        print(f"\n📌 Etapa: {alvo.upper()}")

        modelo, encoder, scaler, features = carregar_artefatos(alvo)

        # Usa apenas as features corretas e garante nomes certos
        dados_para_escalar = dados[features].copy()
        dados_escalados = pd.DataFrame(
            scaler.transform(dados_para_escalar),
            columns=features
        )

        # Predição
        pred_codificada = modelo.predict(dados_escalados)
        pred_nome = encoder.inverse_transform(pred_codificada)

        for i, nome in enumerate(pred_nome):
            print(f"🔸 Amostra {i+1} → {alvo}: {nome}")

        historico_preds[alvo] = pred_nome
        dados[f"pred_{alvo}"] = pred_codificada

    print("\n✅ Predição finalizada com sucesso!")

    # ================= CONSISTÊNCIA HIERÁRQUICA =================
    print("\n🔎 Validando consistência hierárquica das predições...\n")
    resultados_df = pd.DataFrame(historico_preds).reset_index(drop=True)

    try:
        df_completo = pd.concat([dados_input.reset_index(drop=True), resultados_df], axis=1)

        inconsistencias = []

        for i, row in df_completo.iterrows():
            pred_nome = row["nome_cientifico"]
            linha_real = df_completo[df_completo["nome_cientifico"] == pred_nome].iloc[0]

            for nivel in ["classe", "ordem", "familia", "genero"]:
                if row[nivel] != linha_real[nivel]:
                    inconsistencias.append({
                        "amostra": i+1,
                        "nome_cientifico": pred_nome,
                        "nivel": nivel,
                        "esperado": linha_real[nivel],
                        "previsto": row[nivel]
                    })

        if inconsistencias:
            print("⚠️ Inconsistências encontradas entre os níveis taxonômicos previstos:")
            for inc in inconsistencias:
                print(f"🔸 Amostra {inc['amostra']} – {inc['nivel'].capitalize()} incorreta para '{inc['nome_cientifico']}': Previsto '{inc['previsto']}', Esperado '{inc['esperado']}'")
        else:
            print("✅ Todas as predições seguem a hierarquia corretamente.")

    except Exception as e:
        print(f"❌ Erro na verificação de consistência: {e}")

    return resultados_df



# ========== TESTE COM UMA AMOSTRA ==========

if __name__ == "__main__":
    from models.utils_model import carregar_dados

    # Carrega dados
    df = carregar_dados()

    # Seleciona amostras para predição
    amostras_para_predizer = df.sample(100, random_state=42)

    print("🧪 Rodando pipeline para amostras:\n")
    print(amostras_para_predizer[colunas_features])

    # Predição em cascata
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

        # Matriz de confusão
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

        # Comparativo real x previsto
        comparativo = pd.DataFrame({
            'Real': y_true,
            'Previsto': y_pred
        })
        print("\n📌 Comparativo real x previsto:")
        print(comparativo)



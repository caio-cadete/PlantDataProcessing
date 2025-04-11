import os
import joblib
import numpy as np
import pandas as pd
<<<<<<< HEAD
import matplotlib.pyplot as plt
import seaborn as sns

=======
import matplotlib
matplotlib.use("Agg")  # Usa backend sem inter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import joblib
>>>>>>> mathmodels
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
    
<<<<<<< HEAD
    dados = dados_input[colunas_features].copy()
=======
    dados = dados_input.copy().reset_index(drop=True)
>>>>>>> mathmodels
    historico_preds = {}

    for alvo in ALVOS:
        print(f"\n📌 Etapa: {alvo.upper()}")

<<<<<<< HEAD
        # Carrega os artefatos do modelo
        modelo, encoder, scaler = carregar_artefatos(alvo)

        # Garante que só as features corretas sejam usadas
        dados_para_escalar = dados[colunas_features]
        dados_escalados = scaler.transform(dados_para_escalar)
=======
        modelo, encoder, scaler, features = carregar_artefatos(alvo)

        # Usa apenas as features corretas e garante nomes certos
        dados_para_escalar = dados[features].copy()
        dados_escalados = pd.DataFrame(
            scaler.transform(dados_para_escalar),
            columns=features
        )
>>>>>>> mathmodels

        # Predição
        pred_codificada = modelo.predict(dados_escalados)
        pred_nome = encoder.inverse_transform(pred_codificada)

<<<<<<< HEAD
        # Feedback para o usuário
        for i, nome in enumerate(pred_nome):
            print(f"🔸 Amostra {i+1} → {alvo}: {nome}")

        # Armazena a predição
        historico_preds[alvo] = pred_nome

        # Adiciona como nova feature para a próxima rodada (opcional)
        dados[f"pred_{alvo}"] = pred_codificada

    print("\n✅ Predição finalizada com sucesso!")
    return pd.DataFrame(historico_preds)

=======
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
            linha_real = dados_input.iloc[i]
            pred_nome = row["nome_cientifico"]

            for nivel in ["classe", "ordem", "familia", "genero"]:
                if str(row[nivel]) != str(linha_real[nivel]):
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
>>>>>>> mathmodels

# ========== TESTE COM UMA AMOSTRA ==========

if __name__ == "__main__":
    from models.utils_model import carregar_dados

<<<<<<< HEAD
    # Carrega dados e seleciona uma amostra
    df = carregar_dados()
    amostras_para_predizer = df.sample(30, random_state=42)  # <-- Pode aumentar para avaliação mais completa
=======
    # Carrega dados
    df = carregar_dados()

    # Seleciona amostras para predição
    amostras_para_predizer = df.sample(30, random_state=42)

    # 🧠 Adiciona a coluna 'cluster_geo' usando KMeans
    from models.train_model import adicionar_cluster_geo  # Certifique-se de que a função está lá
    amostras_para_predizer = adicionar_cluster_geo(amostras_para_predizer)
>>>>>>> mathmodels

    print("🧪 Rodando pipeline para amostras:\n")
    print(amostras_para_predizer[colunas_features])

<<<<<<< HEAD
=======
    # Predição em cascata
>>>>>>> mathmodels
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

<<<<<<< HEAD
        # Matriz de confusão (plot simples via seaborn)
=======
        # Matriz de confusão
>>>>>>> mathmodels
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
<<<<<<< HEAD
            plt.show()
        except Exception as e:
            print(f"⚠️ Erro ao gerar matriz de confusão: {e}")

        # Comparativo direto
=======
            plt.savefig("models/matriz_confusao_nome_cientifico.png")
            # salva a imagem
            plt.close()  # fecha a figura para liberar memória
        except Exception as e:
            print(f"⚠️ Erro ao gerar matriz de confusão: {e}")

        # Comparativo real x previsto
>>>>>>> mathmodels
        comparativo = pd.DataFrame({
            'Real': y_true,
            'Previsto': y_pred
        })
        print("\n📌 Comparativo real x previsto:")
        print(comparativo)
<<<<<<< HEAD
=======


>>>>>>> mathmodels

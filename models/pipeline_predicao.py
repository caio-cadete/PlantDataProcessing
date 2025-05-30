import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Usa backend sem inter
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gc  # Added garbage collection
import sys
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Features usadas nos modelos
from models.utils_model import colunas_features

# Caminho da pasta com os modelos
PASTA_MODELOS = "trained-models"

# Ordem dos alvos para predição em cascata
ALVOS = ['classe', 'ordem', 'familia', 'genero', 'nome_cientifico']


def limpar_memoria_agressiva():
    """Força limpeza agressiva de memória"""
    gc.collect()
    gc.collect()
    gc.collect()


def carregar_artefatos(target):
    print(f"🔍 Carregando modelo, encoder e scaler para: {target}")
    try:
        # Para o modelo maior, use memory mapping se disponível
        if target == 'nome_cientifico':
            print("⚠️ Carregando modelo grande com otimizações especiais...")
            # Força limpeza antes de carregar o modelo grande
            limpar_memoria_agressiva()
        
        modelo = joblib.load(os.path.join(PASTA_MODELOS, f"{target}_model.pkl"))
        encoder = joblib.load(os.path.join(PASTA_MODELOS, f"{target}_label_encoder.pkl"))
        scaler = joblib.load(os.path.join(PASTA_MODELOS, f"{target}_scaler.pkl"))

        with open(os.path.join(PASTA_MODELOS, f"{target}_features.txt")) as f:
            features = f.read().splitlines()

        return modelo, encoder, scaler, features
    except Exception as e:
        print(f"❌ Erro ao carregar artefatos para '{target}': {e}")
        raise


def predizer_nivel_unico(dados_input, alvo):
    """Prediz um único nível taxonômico e libera a memória imediatamente"""
    print(f"\n📌 Etapa: {alvo.upper()}")
    
    # Limpeza prévia de memória
    limpar_memoria_agressiva()
    
    modelo = None
    encoder = None
    scaler = None
    
    try:
        # Carrega apenas os artefatos necessários para este nível
        modelo, encoder, scaler, features = carregar_artefatos(alvo)
        
        # Usa apenas as features corretas e garante nomes certos
        dados_para_escalar = dados_input[features].copy()
        dados_escalados = pd.DataFrame(
            scaler.transform(dados_para_escalar),
            columns=features
        )

        # Predição
        pred_codificada = modelo.predict(dados_escalados)
        pred_nome = encoder.inverse_transform(pred_codificada)

        for i, nome in enumerate(pred_nome):
            print(f"🔸 Amostra {i+1} → {alvo}: {nome}")

        # Retorna a predição e os códigos
        return pred_nome, pred_codificada
        
    except Exception as e:
        print(f"❌ Erro na predição para {alvo}: {e}")
        raise
    finally:
        # Força a liberação da memória de forma agressiva
        if modelo is not None:
            del modelo
        if encoder is not None:
            del encoder
        if scaler is not None:
            del scaler
        
        # Múltiplas chamadas de garbage collection para garantir limpeza
        limpar_memoria_agressiva()
        print(f"✅ Memória liberada para {alvo}")


def predizer_em_cascata(dados_input):
    print("\n🚀 Iniciando predição em cascata...\n")
    
    # Limpeza inicial de memória
    limpar_memoria_agressiva()
    
    dados = dados_input.copy().reset_index(drop=True)
    historico_preds = {}

    for alvo in ALVOS:
        try:
            # Prediz um nível por vez e libera a memória
            pred_nome, pred_codificada = predizer_nivel_unico(dados, alvo)
            
            historico_preds[alvo] = pred_nome
            dados[f"pred_{alvo}"] = pred_codificada
            
            # Limpeza após cada predição
            limpar_memoria_agressiva()
            
        except Exception as e:
            print(f"❌ Erro na predição em cascata para {alvo}: {e}")
            # Mesmo em caso de erro, tenta limpar a memória
            limpar_memoria_agressiva()
            raise

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

# ========== TESTE COM UMA AMOSTRA ==========

if __name__ == "__main__":
    from models.utils_model import carregar_dados

    # Carrega dados
    df = carregar_dados()

    # Seleciona amostras para predição
    amostras_para_predizer = df.sample(30, random_state=42)

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
            plt.savefig("trained-models/matriz_confusao_nome_cientifico.png")
            # salva a imagem
            plt.close()  # fecha a figura para liberar memória
        except Exception as e:
            print(f"⚠️ Erro ao gerar matriz de confusão: {e}")

        # Comparativo real x previsto
        comparativo = pd.DataFrame({
            'Real': y_true,
            'Previsto': y_pred
        })
        print("\n📌 Comparativo real x previsto:")
        print(comparativo)



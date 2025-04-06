# models/utils_model.py
import pandas as pd

colunas_features = [
    "latitude", 
    "longitude", 
    "Temperatura_Media_Anual_GrausC", 
    "Precipitacao_Media_Anual_mm", 
    # "Temperatura_Media_Janeiro_GrausC", 
    # "Temperatura_Media_Fevereiro_GrausC", 
    # "Temperatura_Media_Marco_GrausC", 
    # "Temperatura_Media_Abril_GrausC", 
    # "Temperatura_Media_Maio_GrausC", 
    # "Temperatura_Media_Junho_GrausC", 
    # "Temperatura_Media_Julho_GrausC", 
    # "Temperatura_Media_Agosto_GrausC", 
    # "Temperatura_Media_Setembro_GrausC", 
    # "Temperatura_Media_Outubro_GrausC", 
    # "Temperatura_Media_Novembro_GrausC", 
    # "Temperatura_Media_Dezembro_GrausC", 
    # "Precipitacao_Media_Janeiro_mm", 
    # "Precipitacao_Media_Fevereiro_mm", 
    # "Precipitacao_Media_Marco_mm", 
    # "Precipitacao_Media_Abril_mm", 
    # "Precipitacao_Media_Maio_mm", 
    # "Precipitacao_Media_Junho_mm", 
    # "Precipitacao_Media_Julho_mm", 
    # "Precipitacao_Media_Agosto_mm", 
    # "Precipitacao_Media_Setembro_mm", 
    # "Precipitacao_Media_Outubro_mm", 
    # "Precipitacao_Media_Novembro_mm", 
    # "Precipitacao_Media_Dezembro_mm", 
    # "Capacidade_Armazenamento_Agua_mm", 
    # "Teor_Argila_Subsolo_%", 
    # "Teor_Cascalho_Subsolo_%", 
    # "Teor_Carbono_Organico_Subsolo_%", 
    # "Teor_Areia_Subsolo_%", 
    # "Teor_Silte_Subsolo_%", 
    # "Teor_Argila_Cam_Superior_%", 
    # "Teor_Cascalho_Cam_Superior_%", 
    # "Teor_Carbono_Organico_Cam_Superior_%", 
    # "Teor_Areia_Cam_Superior_%", 
    # "Teor_Silte_Cam_Superior_%", 
    "Isotermalidade", 
    "Sazonalidade_Temperatura"
]

def carregar_dados(path="data/plantas_clima_rj_processado.csv"):
    return pd.read_csv(path, sep=';', encoding='latin1', low_memory=False)

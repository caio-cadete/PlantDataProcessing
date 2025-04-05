# models/utils_model.py
import pandas as pd

colunas_features = [
    'latitude', 'longitude',
    'TempAnnMean_DegC', 'PrecipAnnMean_mm',
    'S_Clay_%', 'S_Sand_%', 'S_Silt_%', 'S_OC_%',
    'T_Clay_%', 'T_Sand_%', 'T_Silt_%', 'T_OC_%',
    'Isothermality', 'TemperatureSeasonality'
]

def carregar_dados(path="data/plantas_clima_rj_processado.csv"):
    return pd.read_csv(path, sep=';', encoding='latin1', low_memory=False)

# app/utils_app.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.utils_model import colunas_features, carregar_dados

import pandas as pd


def carregar_dados_processados():
    df = carregar_dados()
    return df.dropna(subset=colunas_features + ['nome_cientifico'])

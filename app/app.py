import streamlit as st
import joblib
import pandas as pd
import numpy as np
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from utils_app import carregar_dados_processados, colunas_features
import time
import os
import logging
from pathlib import Path
from datetime import datetime

# -------------------- Logs --------------------
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------- Cache --------------------
@st.cache_data
def carregar_modelo_e_encoder(alvo):
    modelo = joblib.load(f"models/modelo_{alvo}.pkl")
    encoder = joblib.load(f"models/label_encoder_{alvo}.pkl")
    return modelo, encoder

@st.cache_data
def carregar_dados():
    return carregar_dados_processados()

# -------------------- Dados e Modelos --------------------
df = carregar_dados()

alvos = ['classe', 'ordem', 'familia', 'genero', 'nome_cientifico']

if "modelos" not in st.session_state or "encoders" not in st.session_state:
    st.session_state.modelos = {}
    st.session_state.encoders = {}
    with st.spinner("🔄 Carregando modelos em cascata..."):
        for alvo in alvos:
            st.session_state.modelos[alvo], st.session_state.encoders[alvo] = carregar_modelo_e_encoder(alvo)
        st.success("✅ Modelos carregados com sucesso!")

# Recupera os modelos e encoders já carregados do session_state
modelos = st.session_state.modelos
encoders = st.session_state.encoders

# -------------------- Mapa --------------------
st.markdown("🖱️ **Desenhe um retângulo no mapa abaixo** para selecionar a área de análise.")

# Criação do mapa
max_bounds = [[-24.0, -44.5], [-20.5, -40.5]]
mapa = folium.Map(location=[-22.5, -43.5], zoom_start=8, tiles="CartoDB positron", min_zoom=7, max_zoom=18, control_scale=True, prefer_canvas=True)

mapa.fit_bounds(max_bounds)
mapa.options['maxBounds'] = max_bounds

# Adiciona controle de desenho
Draw(
    export=True,
    draw_options={
        'polyline': False, 'polygon': False, 'circle': False,
        'marker': False, 'circlemarker': False, 'rectangle': True
    }
).add_to(mapa)

# Exibe o mapa e captura a seleção
output = st_folium(mapa, width=1000, height=700, key="mapa_interativo")


st.markdown("🔍 **Área selecionada:**")

def expandir_area(lon_min, lon_max, lat_min, lat_max, delta):
    return lon_min - delta, lon_max + delta, lat_min - delta, lat_max + delta

def predizer_em_cascata(entrada, modelos, encoders, alvos):
    resultados = {}
    # Começa com apenas as features usadas no modelo
    entrada_temp = entrada[colunas_features + ['latitude', 'longitude']].copy()
    for alvo in alvos:
        modelo_path = Path(f"models/modelo_{alvo}.pkl")
        encoder_path = Path(f"models/label_encoder_{alvo}.pkl")
        if not modelo_path.exists() or not encoder_path.exists():
            st.error(f"Modelo ou encoder para '{alvo}' não encontrado.")
            st.stop()
        modelo = modelos[alvo]
        encoder = encoders[alvo]
        entrada_temp = entrada[colunas_features + ['latitude', 'longitude']].copy()
        for nivel in resultados:
            entrada_temp[nivel] = encoders[nivel].transform([resultados[nivel]])
        pred_cod = modelo.predict(entrada_temp)[0]
        pred_str = encoder.inverse_transform([pred_cod])[0]
        resultados[alvo] = pred_str
        entrada_temp[alvo] = pred_cod
    return resultados

# Tenta buscar dados inicialmente na área exata
# Função para tentar selecionar dados com expansão progressiva
def selecionar_dados_proximos(df, lat_min, lat_max, lon_min, lon_max, deltas):
    for delta in deltas:
        lon_min_exp, lon_max_exp, lat_min_exp, lat_max_exp = expandir_area(
            lon_min, lon_max, lat_min, lat_max, delta
        )
        df_filtrado = df[
            (df['latitude'].between(lat_min_exp, lat_max_exp)) &
            (df['longitude'].between(lon_min_exp, lon_max_exp))
        ]
        if not df_filtrado.empty:
            st.info(f"🔄 Nenhum dado exato na área. Usando dados próximos em um raio de ±{delta:.2f}° (~{delta * 111:.1f} km).")
            return df_filtrado
    return None

# Processa a nova seleção, se houver
if output and output.get("last_active_drawing"):
    coords = output["last_active_drawing"]["geometry"]["coordinates"][0]
    lon_min = min(c[0] for c in coords)
    lon_max = max(c[0] for c in coords)
    lat_min = min(c[1] for c in coords)
    lat_max = max(c[1] for c in coords)

    st.success(f"Área selecionada: de ({lat_min:.2f}, {lon_min:.2f}) até ({lat_max:.2f}, {lon_max:.2f})")

    # Primeiro filtro exato na área desenhada
    df_selecionado = df[
        (df['latitude'].between(lat_min, lat_max)) &
        (df['longitude'].between(lon_min, lon_max))
    ]

    # Se vazio, tenta encontrar dados nas proximidades
    if df_selecionado.empty:
        deltas = [0.02, 0.05, 0.1]  # Expansões progressivas em graus
        df_proximo = selecionar_dados_proximos(df, lat_min, lat_max, lon_min, lon_max, deltas)

        if df_proximo is not None:
            df_selecionado = df_proximo
        else:
            st.warning("⚠️ Nenhum dado encontrado, mesmo com expansão. Usando média geral do estado.")
            df_selecionado = df.copy()
            lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
            lon_min, lon_max = df['longitude'].min(), df['longitude'].max()

    # Define ponto central (mesmo que tenha sido expandido)
    lat_centro = (lat_min + lat_max) / 2
    lon_centro = (lon_min + lon_max) / 2

    # Cria entrada com média das features da região
    entrada = df_selecionado[colunas_features].mean().to_frame().T
    entrada['latitude'] = lat_centro
    entrada['longitude'] = lon_centro

    # Aqui você chama a predição com a entrada
    try:
        resultados = predizer_em_cascata(entrada, modelos, encoders, alvos)
        nome_predito = resultados['nome_cientifico']

        linha_planta = df[df['nome_cientifico'].str.strip().str.lower() == nome_predito.strip().lower()]
        nome_popular_series = linha_planta['nome_popular'].dropna()

        if not nome_popular_series.empty:
            nome_popular = nome_popular_series.mode()[0]
            tooltip = f"{nome_popular} ({nome_predito})"
            st.success(f"🌿 Planta recomendada: **{nome_popular}** ({nome_predito})")
        else:
            tooltip = nome_predito
            st.success(f"🌿 Planta recomendada: **{nome_predito}**")

        st.session_state.retangulos.append({
            "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
            "tooltip": tooltip
        })

    except Exception as e:
        st.error(f"❌ Erro durante a predição: {str(e)}")
        logging.error(f"Erro na predição em cascata: {str(e)}")

else:
    st.info("🖱️ Por favor, selecione uma área desenhando um retângulo no mapa acima.")

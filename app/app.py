import streamlit as st
import joblib
import pandas as pd
import numpy as np
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import os
import logging
from pathlib import Path
from datetime import datetime
from utils_app import carregar_dados_processados, colunas_features
# -------------------- Logs --------------------
if "retangulos" not in st.session_state:
    st.session_state.retangulos = []

os.makedirs("logs", exist_ok=True)
log_filename = f"logs/log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------- Cache --------------------
@st.cache_data
def carregar_modelo_e_encoder(alvo):
    modelo = joblib.load(f"models/modelo_{alvo}.pkl")
    encoder = joblib.load(f"models/label_encoder_{alvo}.pkl")
    scaler = joblib.load(f"models/scaler_{alvo}.pkl")  
    return modelo, encoder, scaler

@st.cache_data
def carregar_dados():
    return carregar_dados_processados()

# -------------------- Dados e Modelos --------------------
df = carregar_dados()

alvos = ['classe', 'ordem', 'familia', 'genero', 'nome_cientifico']

if "modelos" not in st.session_state or "encoders" not in st.session_state or "scalers" not in st.session_state:
    st.session_state.modelos = {}
    st.session_state.encoders = {}
    st.session_state.scalers = {}  # Adiciona o dicionário para armazenar os scalers
    with st.spinner("🔄 Carregando modelos em cascata..."):
        for alvo in alvos:
            modelo, encoder, scaler = carregar_modelo_e_encoder(alvo)
            st.session_state.modelos[alvo] = modelo
            st.session_state.encoders[alvo] = encoder
            st.session_state.scalers[alvo] = scaler  # Armazena o scaler
        st.success("✅ Modelos e scalers carregados com sucesso!")


modelos = st.session_state.modelos
encoders = st.session_state.encoders
scalers = st.session_state.scalers 
# -------------------- Mapa --------------------
st.markdown("🖱️ **Desenhe um retângulo no mapa abaixo** para selecionar a área de análise.")

max_bounds = [[-24.0, -44.5], [-20.5, -40.5]]
mapa = folium.Map(location=[-22.5, -43.5], zoom_start=8, min_zoom=7, max_zoom=18, control_scale=True, prefer_canvas=True)
mapa.fit_bounds(max_bounds)
mapa.options['maxBounds'] = max_bounds

# Adiciona controle de desenho
Draw(
    export=True,
    draw_options={'polyline': False, 'polygon': False, 'circle': False, 'marker': False, 'circlemarker': False, 'rectangle': True}
).add_to(mapa)

output = st_folium(mapa, width=1000, height=700, key="mapa_interativo")
st.markdown("🔍 **Área selecionada:**")


def predizer_em_cascata(entrada, modelos, encoders, alvos, scalers):
    resultados = {}
    entrada_temp = entrada.copy()

    for alvo in alvos:
        modelo = modelos[alvo]
        encoder = encoders[alvo]
        scaler = scalers[alvo]

        # Reindexa os dados de entrada com base nas colunas de features
        entrada_modelo = entrada_temp.reindex(columns=colunas_features)

        # Garante que os dados de entrada estejam escalonados de acordo com o scaler
        entrada_modelo = scaler.transform(entrada_modelo)

        # Realiza a predição
        pred_cod = modelo.predict(entrada_modelo)[0]
        pred_str = encoder.inverse_transform([pred_cod])[0]

        resultados[alvo] = pred_str
        entrada_temp[alvo] = pred_cod

    return resultados



# Processa a nova seleção, se houver
if output and output.get("last_active_drawing"):
    coords = output["last_active_drawing"]["geometry"]["coordinates"][0]
    lon_min = min(c[0] for c in coords)
    lon_max = max(c[0] for c in coords)
    lat_min = min(c[1] for c in coords)
    lat_max = max(c[1] for c in coords)

    st.success(f"Área selecionada: de ({lat_min:.2f}, {lon_min:.2f}) até ({lat_max:.2f}, {lon_max:.2f})")

    entrada = pd.DataFrame([{
        'latitude': (lat_min + lat_max) / 2,
        'longitude': (lon_min + lon_max) / 2,
    }])

    try:
        # Chame a função com todos os parâmetros necessários, incluindo 'scalers'
        resultados = predizer_em_cascata(entrada, modelos, encoders, alvos, scalers)


        if isinstance(resultados, dict):
            nome_predito = resultados['nome_cientifico']
        else:
            st.error("❌ Erro: resultados não são um dicionário.")


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

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
from datetime import datetime

# -------------------- Logs --------------------
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------- Cache --------------------
@st.cache_data
def carregar_modelo():
    return joblib.load('modelo_random_forest.pkl')

@st.cache_data
def carregar_encoder():
    return joblib.load('label_encoder.pkl')

@st.cache_data
def carregar_dados():
    return carregar_dados_processados()

# -------------------- Título --------------------
st.title("🌱 Preditor de Planta Ideal para Reflorestamento no RJ")

# -------------------- Carregamento --------------------
if 'modelo' not in st.session_state or 'encoder' not in st.session_state or 'df' not in st.session_state:
    with st.spinner("🔄 Carregando modelo, encoder e dados..."):
        st.session_state.modelo = carregar_modelo()
        st.session_state.encoder = carregar_encoder()
        st.session_state.df = carregar_dados()
    st.success(f"✅ Dados carregados! Total de registros: {len(st.session_state.df)}")
else:
    modelo = st.session_state.modelo
    le = st.session_state.encoder
    df = st.session_state.df

# -------------------- Mapa único com seleção e predição --------------------
st.markdown("🖱️ **Desenhe um retângulo no mapa abaixo** para selecionar a área de análise.")

# Limites aproximados do estado do RJ
max_bounds = [[-24.0, -44.5], [-20.5, -40.5]]

# Criando o mapa com o zoom inicial e limitando o zoom para o RJ
mapa = folium.Map(
    location=[-22.5, -43.5],
    zoom_start=8,
    min_zoom=7,  # Limita o zoom out para a área do RJ
    max_zoom=18,  # Permite o zoom in até um nível máximo
    control_scale=True
)

# Ajusta os limites do mapa para o Rio de Janeiro e fixa o zoom
mapa.fit_bounds(max_bounds)
mapa.options['maxBounds'] = max_bounds

Draw(
    export=True,
    draw_options={
        'polyline': False, 'polygon': False, 'circle': False,
        'marker': False, 'circlemarker': False, 'rectangle': True
    }
).add_to(mapa)

output = st_folium(mapa, width=1000, height=700)

if output and output.get("last_active_drawing"):
    coords = output["last_active_drawing"]["geometry"]["coordinates"][0]
    lon_min = min(c[0] for c in coords)
    lon_max = max(c[0] for c in coords)
    lat_min = min(c[1] for c in coords)
    lat_max = max(c[1] for c in coords)

    st.success(f"Área selecionada: de ({lat_min:.2f}, {lon_min:.2f}) até ({lat_max:.2f}, {lon_max:.2f})")

    df_selecionado = df[
        (df['latitude'].between(lat_min, lat_max)) &
        (df['longitude'].between(lon_min, lon_max))
    ]

    if df_selecionado.empty:
        st.warning("⚠️ Nenhum dado encontrado dentro da área selecionada.")
    else:
        entrada = df_selecionado[colunas_features].mean().to_frame().T
        entrada['latitude'] = (lat_min + lat_max) / 2
        entrada['longitude'] = (lon_min + lon_max) / 2

        try:
            pred_cod = modelo.predict(entrada)[0]
            pred_nome = le.inverse_transform([pred_cod])[0]
            nome_popular = df[df['nome_cientifico'] == pred_nome]['nome_popular'].mode()[0]
            tooltip = f"{nome_popular} ({pred_nome})"

            folium.Rectangle(
                bounds=[[lat_min, lon_min], [lat_max, lon_max]],
                color="green",
                fill=True,
                fill_opacity=0.4,
                tooltip=tooltip
            ).add_to(mapa)
        
            st.success(f"🌿 Planta recomendada: **{nome_popular}** ({pred_nome})")
        except ValueError as ve:
            st.error(f"❌ Erro de valor na predição: {str(ve)}")
        except KeyError as ke:
            st.error(f"❌ Erro de chave na predição: {str(ke)}")
        except Exception as e:
            st.error(f"❌ Erro na predição: {str(e)}")
            logging.error(f"Erro na predição: {str(e)}")
    st.warning("⚠️ Desenhe uma área no mapa acima para iniciar a análise.")

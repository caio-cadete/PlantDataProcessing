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
def carregar_modelo_e_encoder(alvo):
    modelo = joblib.load(f"models/modelo_{alvo}.pkl")
    encoder = joblib.load(f"models/label_encoder_{alvo}.pkl")
    return modelo, encoder

@st.cache_data
def carregar_dados():
    return carregar_dados_processados()

# -------------------- Título --------------------
st.title("🌱 Preditor de Planta Ideal para Reflorestamento no RJ (Método Cascata)")

# -------------------- Dados e Modelos --------------------
df = carregar_dados()

modelos = {}
encoders = {}
alvos = ['classe', 'ordem', 'familia', 'genero', 'nome_cientifico']

with st.spinner("🔄 Carregando modelos em cascata..."):
    for alvo in alvos:
        modelos[alvo], encoders[alvo] = carregar_modelo_e_encoder(alvo)
    st.success("✅ Modelos carregados com sucesso!")

# -------------------- Mapa --------------------
st.markdown("🖱️ **Desenhe um retângulo no mapa abaixo** para selecionar a área de análise.")

# Inicializa os retângulos salvos na sessão
if "retangulos" not in st.session_state:
    st.session_state.retangulos = []

# Criação do mapa
max_bounds = [[-24.0, -44.5], [-20.5, -40.5]]
mapa = folium.Map(location=[-22.5, -43.5], zoom_start=8, min_zoom=7, max_zoom=18, control_scale=True)
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

# Adiciona os retângulos anteriores salvos no estado
for r in st.session_state.retangulos:
    folium.Rectangle(
        bounds=r["bounds"],
        color="green",
        fill=True,
        fill_opacity=0.4,
        tooltip=r["tooltip"]
    ).add_to(mapa)

# Exibe o mapa e captura a seleção
output = st_folium(mapa, width=1000, height=700, key="mapa_interativo")
st.markdown("🔍 **Área selecionada:**")

# Processa a nova seleção, se houver
# Processa a nova seleção, se houver
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
        # Criação da entrada com os dados médios
        entrada = df_selecionado[colunas_features].mean().to_frame().T
        entrada['latitude'] = (lat_min + lat_max) / 2
        entrada['longitude'] = (lon_min + lon_max) / 2

        try:
            resultados = {}
            for alvo in alvos:
                modelo = modelos[alvo]
                encoder = encoders[alvo]

                entrada_temp = entrada.copy()

                # AQUI APLICAMOS A TRANSFORMAÇÃO NAS VARIÁVEIS CATEGÓRICAS
                for feature in ['classe', 'ordem', 'familia', 'genero', 'nome_cientifico']:
                    if feature in entrada_temp.columns:
                        entrada_temp[feature] = encoder.transform(entrada_temp[feature])

                # Predição com os dados já codificados
                pred_cod = modelo.predict(entrada_temp)[0]
                pred_str = encoder.inverse_transform([pred_cod])[0]
                resultados[alvo] = pred_str

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

            # Adiciona esse retângulo ao session_state
            st.session_state.retangulos.append({
                "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
                "tooltip": tooltip
            })

        except Exception as e:
            st.error(f"❌ Erro durante a predição: {str(e)}")
            logging.error(f"Erro na predição em cascata: {str(e)}")

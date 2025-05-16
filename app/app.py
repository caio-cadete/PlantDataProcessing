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

# -------------------- Função Haversine --------------------
def haversine(lat1, lon1, lat2, lon2):
    """
    Calcula a distância em km entre dois pontos geográficos usando a fórmula de Haversine.
    """
    R = 6371  # Raio da Terra em km
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad)*np.cos(lat2_rad)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

# -------------------- Dados e Modelos --------------------
df = carregar_dados()

alvos = ['classe', 'ordem', 'familia', 'genero', 'nome_cientifico']

if "modelos" not in st.session_state or "encoders" not in st.session_state or "scalers" not in st.session_state:
    st.session_state.modelos = {}
    st.session_state.encoders = {}
    st.session_state.scalers = {}
    with st.spinner("🔄 Carregando modelos em cascata..."):
        for alvo in alvos:
            modelo, encoder, scaler = carregar_modelo_e_encoder(alvo)
            st.session_state.modelos[alvo] = modelo
            st.session_state.encoders[alvo] = encoder
            st.session_state.scalers[alvo] = scaler
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

Draw(
    export=False,
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

        colunas_modelo = modelo.feature_names_in_
        entrada_modelo = entrada_temp.reindex(columns=colunas_modelo)
        entrada_modelo = scaler.transform(entrada_modelo)
        entrada_modelo = pd.DataFrame(entrada_modelo, columns=colunas_modelo)

        pred_cod = modelo.predict(entrada_modelo)[0]
        pred_str = encoder.inverse_transform([pred_cod])[0]

        resultados[alvo] = pred_str
        entrada_temp[alvo] = pred_cod

    for feature in colunas_features:
        if feature in entrada_temp.columns:
            resultados[feature] = entrada_temp.iloc[0][feature]

    return resultados

def buscar_dados_proximos(df, centro_lat, centro_lon, raio_max_km=5, passo_km=0.5):
    """
    Busca dados no DataFrame dentro de um raio crescente a partir do centro.
    Retorna os dados encontrados dentro do primeiro raio que contenha alguma amostra, ou None se nada encontrado até o raio máximo.
    """
    raio = passo_km
    while raio <= raio_max_km:
        df['distancia'] = haversine(centro_lat, centro_lon, df['latitude'], df['longitude'])
        dados_proximos = df[df['distancia'] <= raio]
        if not dados_proximos.empty:
            df.drop(columns=['distancia'], inplace=True)
            return dados_proximos
        raio += passo_km
    df.drop(columns=['distancia'], inplace=True)
    return None

if output and output.get("last_active_drawing"):
    coords = output["last_active_drawing"]["geometry"]["coordinates"][0]
    lon_min = min(c[0] for c in coords)
    lon_max = max(c[0] for c in coords)
    lat_min = min(c[1] for c in coords)
    lat_max = max(c[1] for c in coords)

    st.success(f"Área selecionada: de ({lat_min:.4f}, {lon_min:.4f}) até ({lat_max:.4f}, {lon_max:.4f})")

    # Filtra as amostras dentro do retângulo
    dados_area = df[(df['latitude'] >= lat_min) & (df['latitude'] <= lat_max) & (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max)]

    if not dados_area.empty:
        entrada = dados_area[colunas_features].mean().to_frame().T
        entrada['latitude'] = (lat_min + lat_max) / 2
        entrada['longitude'] = (lon_min + lon_max) / 2

        try:
            resultados = predizer_em_cascata(entrada, modelos, encoders, alvos, scalers)
            nome_predito = resultados.get('nome_cientifico')

            if nome_predito:
                linha_planta = df[df['nome_cientifico'].str.strip().str.lower() == nome_predito.strip().lower()]
                nome_popular_series = linha_planta['nome_popular'].dropna()

                if not nome_popular_series.empty:
                    nome_popular = nome_popular_series.mode()[0]
                    tooltip = f"{nome_popular} ({nome_predito})"
                    st.success(f"🌿 Planta recomendada: **{nome_popular}** ({nome_predito})")
                else:
                    tooltip = nome_predito
                    st.success(f"🌿 Planta recomendada: **{nome_predito}**")

                st.markdown("### Variáveis de entrada usadas na predição:")
                for feature in colunas_features:
                    if feature in resultados:
                        st.write(f"**{feature}**: {resultados[feature]}")

                st.session_state.retangulos.append({
                    "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
                    "tooltip": tooltip
                })
            else:
                st.warning("⚠️ Não foi possível identificar o nome científico da planta.")

        except Exception as e:
            st.error(f"❌ Erro durante a predição: {str(e)}")
            logging.error(f"Erro na predição em cascata: {str(e)}")

    else:
        # Sem dados no retângulo, tenta buscar dados próximos num raio crescente
        centro_lat = (lat_min + lat_max) / 2
        centro_lon = (lon_min + lon_max) / 2
        dados_proximos = buscar_dados_proximos(df, centro_lat, centro_lon)

        if dados_proximos is not None:
            st.warning("ℹ️ A área selecionada não contém dados. Utilizando informações de locais próximos para gerar a previsão.")
            entrada = dados_proximos[colunas_features].mean().to_frame().T
            entrada['latitude'] = centro_lat
            entrada['longitude'] = centro_lon

            try:
                resultados = predizer_em_cascata(entrada, modelos, encoders, alvos, scalers)
                nome_predito = resultados.get('nome_cientifico')

                if nome_predito:
                    linha_planta = df[df['nome_cientifico'].str.strip().str.lower() == nome_predito.strip().lower()]
                    nome_popular_series = linha_planta['nome_popular'].dropna()

                    if not nome_popular_series.empty:
                        nome_popular = nome_popular_series.mode()[0]
                        tooltip = f"{nome_popular} ({nome_predito})"
                        st.success(f"🌿 Planta recomendada: **{nome_popular}** ({nome_predito})")
                    else:
                        tooltip = nome_predito
                        st.success(f"🌿 Planta recomendada: **{nome_predito}**")

                    st.markdown("### Variáveis de entrada usadas na predição:")
                    for feature in colunas_features:
                        if feature in resultados:
                            st.write(f"**{feature}**: {resultados[feature]}")

                    st.session_state.retangulos.append({
                        "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
                        "tooltip": tooltip
                    })
                else:
                    st.warning("⚠️ Não foi possível identificar o nome da espécie da planta.")

            except Exception as e:
                st.error(f"❌ Erro durante a predição: {str(e)}")
                logging.error(f"Erro na predição em cascata: {str(e)}")

        else:
            st.warning("⚠️ Nenhum dado encontrado dentro da área selecionada nem nas proximidades.")

else:
    st.info("🖱️ Por favor, selecione uma área desenhando um retângulo no mapa acima.")

import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Carregar os arquivos
df_clima = pd.read_csv('dataset_climasolotemp_rj.csv', sep=';')

# Carregar o arquivo de plantas, com tratamento para vírgula como separador decimal
df_plantas = pd.read_csv('dataset_plantae_rj_min_30.csv', sep=';', encoding='latin1')

# Verificar as colunas
print("Colunas do DataFrame df_plantas:", df_plantas.columns)

# Remover espaços extras nas colunas
df_plantas.columns = df_plantas.columns.str.strip()

# Função para garantir que as latitudes e longitudes sejam convertidas corretamente
def convert_to_float(value):
    try:
        # Substituir vírgula por ponto, se necessário
        value = str(value).replace(',', '.')
        # Limitar a 6 casas decimais para coordenadas
        return round(float(value), 6)
    except ValueError:
        return np.nan  # Retorna NaN se não puder converter

# Aplicar a conversão nas latitudes e longitudes
df_plantas['latitude'] = df_plantas['latitude'].apply(lambda x: convert_to_float(x))
df_plantas['longitude'] = df_plantas['longitude'].apply(lambda x: convert_to_float(x))

# Verificar se há NaNs restantes
print("NaNs em df_plantas após o tratamento:")
print(df_plantas[['latitude', 'longitude']].isna().sum())

# Remover as linhas com NaN nas coordenadas (caso haja)
df_plantas.dropna(subset=['latitude', 'longitude'], inplace=True)

# Extrair coordenadas
plant_coords = df_plantas[['latitude', 'longitude']].to_numpy()
clima_coords = df_clima[['latitude', 'longitude']].to_numpy()

# Treinar modelo de vizinho mais próximo
nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
nn.fit(clima_coords)

# Encontrar o clima mais próximo para cada ponto de planta
distances, indices = nn.kneighbors(plant_coords)

# Selecionar as linhas correspondentes no DataFrame de clima
df_clima_matched = df_clima.iloc[indices.flatten()].reset_index(drop=True)

# Remover colunas duplicadas antes de unir (opcional)
df_clima_matched = df_clima_matched.drop(columns=['latitude', 'longitude'])

# Resetar índice de plantas também
df_plantas = df_plantas.reset_index(drop=True)

# Concatenar os dois DataFrames
df_resultado = pd.concat([df_plantas, df_clima_matched], axis=1)

# Define um limite de distância aceitável (ex: 0.05 graus, que dá ~5km)
dist_max = 0.05
df_resultado['distancia'] = distances.flatten()

# Filtra pares com distância aceitável
df_resultado = df_resultado[df_resultado['distancia'] <= dist_max].drop(columns='distancia')

# Salvar em CSV com as latitudes e longitudes devidamente formatadas
df_resultado.to_csv('dataset_cruzado2.csv', index=False, sep=';', encoding='latin1')

print("Cruzamento concluído! Arquivo salvo como 'dataset_cruzado2.csv'")

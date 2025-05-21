from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from models.pipeline_predicao import predizer_em_cascata
from models.utils_model import colunas_features, carregar_dados

app = FastAPI()

# Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PlantInput(BaseModel):
    latitude: float
    longitude: float
    Average_Annual_Temperature_C: float
    Average_Annual_Precipitation_mm: float
    Average_January_Temperature_C: float
    Average_February_Temperature_C: float
    Average_March_Temperature_C: float
    Average_April_Temperature_C: float
    Average_May_Temperature_C: float
    Average_June_Temperature_C: float
    Average_July_Temperature_C: float
    Average_August_Temperature_C: float
    Average_September_Temperature_C: float
    Average_October_Temperature_C: float
    Average_November_Temperature_C: float
    Average_December_Temperature_C: float
    Average_January_Precipitation_mm: float
    Average_February_Precipitation_mm: float
    Average_March_Precipitation_mm: float
    Average_April_Precipitation_mm: float
    Average_May_Precipitation_mm: float
    Average_June_Precipitation_mm: float
    Average_July_Precipitation_mm: float
    Average_August_Precipitation_mm: float
    Average_September_Precipitation_mm: float
    Average_October_Precipitation_mm: float
    Average_November_Precipitation_mm: float
    Average_December_Precipitation_mm: float
    Water_Storage_Capacity_mm: float
    Isothermality: float
    Temperature_Seasonality: float

class BBoxInput(BaseModel):
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

@app.post("/recommend-plant")
def recommend_plant(input_data: PlantInput):
    # Convert input to DataFrame
    df = pd.DataFrame([input_data.dict()])
    # Run prediction
    result_df = predizer_em_cascata(df)
    # Return the predicted scientific name and all levels
    return result_df.iloc[0].to_dict()

@app.post("/recommend-plant-bbox")
def recommend_plant_bbox(bbox: BBoxInput):
    df = carregar_dados()
    # Filter samples within the rectangle
    area_data = df[(df['latitude'] >= bbox.lat_min) & (df['latitude'] <= bbox.lat_max) & (df['longitude'] >= bbox.lon_min) & (df['longitude'] <= bbox.lon_max)]
    center_lat = (bbox.lat_min + bbox.lat_max) / 2
    center_lon = (bbox.lon_min + bbox.lon_max) / 2

    if not area_data.empty:
        input_data = area_data[colunas_features].mean().to_frame().T
        input_data['latitude'] = center_lat
        input_data['longitude'] = center_lon
    else:
        # Try to find nearest data (expand radius)
        # For simplicity, use the closest point by Euclidean distance
        df['dist'] = ((df['latitude'] - center_lat)**2 + (df['longitude'] - center_lon)**2)**0.5
        nearest = df.nsmallest(10, 'dist')
        if nearest.empty:
            raise HTTPException(status_code=404, detail="No data found in the area or nearby.")
        input_data = nearest[colunas_features].mean().to_frame().T
        input_data['latitude'] = center_lat
        input_data['longitude'] = center_lon

    # Run prediction
    result_df = predizer_em_cascata(input_data)
    result = result_df.iloc[0].to_dict()
    formatted_result = {
        "class": result['classe'],
        "family": result['familia'],
        "order": result['ordem'],
        "gender": result['genero'],
        "cientific_name": result['nome_cientifico'],
    }
    # Also return the features used
    features_used = input_data.iloc[0].to_dict()

    formatted_features_used = {
        "latitude": features_used['latitude'],
        "longitude": features_used['longitude'],
        "anual_mean_temperature": features_used['Temperatura_Media_Anual_GrausC'],
        "anual_mean_precipitation": features_used['Precipitacao_Media_Anual_mm'],
        "isothermality": features_used['Isotermalidade'],
        "temperature_seasonality": features_used['Sazonalidade_Temperatura'],
    }

    return {"prediction": formatted_result, "features_used": formatted_features_used} 
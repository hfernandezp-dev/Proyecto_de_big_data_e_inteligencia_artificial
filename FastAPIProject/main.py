from fastapi import FastAPI, Header, HTTPException, Depends
from pathlib import Path
import pandas as pd
import os
from dotenv import load_dotenv
import joblib
from schemas import CancionEntrada
import logging


def iniciar_logger():

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "API.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

iniciar_logger()

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
MODELOS_DIR = BASE_DIR / "src" / "modelos"
env_path = BASE_DIR / "environment" / ".env"
scaler_path = MODELOS_DIR / "scaler.pkl"
load_dotenv(dotenv_path=env_path)

def creacion_dataset_canciones():
    try:
        clust_feats = ['instrumentalness', 'speechiness', 'danceability', 'valence', 'tempo']
        scaler = joblib.load(scaler_path)
        csv_path_para_clusterizar = BASE_DIR / "src" / "datasets" / "spotify_data.csv"
        modelo_path2 = MODELOS_DIR / "kmeans_spotify_model.pkl"
        df_canciones_para_clusterizar = pd.read_csv(csv_path_para_clusterizar)
        X = df_canciones_para_clusterizar[clust_feats]
        X_scaled = scaler.transform(X)
        kmeans = joblib.load(modelo_path2)
        df_canciones_para_clusterizar['cluster'] = kmeans.predict(X_scaled)
        ruta_dataset_clusterizado = 'datasets/'
        os.makedirs(ruta_dataset_clusterizado, exist_ok=True)
        if df_canciones_para_clusterizar is not None:
            logging.info("Se ha creado el dataset de canciones correctamente")
        df_canciones_para_clusterizar.to_csv("datasets/canciones_clusterizadas.csv", index=False)
    except Exception as e:
        logging.error(f"Error en la creacion del dataset de canciones: {e}")


creacion_dataset_canciones()


#simula la entrada de datos ya que no se ha desplegado
@app.get("/obtener_datos_emul")
async def obtener2_datos_emul(x_api_key: str | None = Header(None, alias="x-api-key")):
    csv_path = BASE_DIR / "src" / "datasets" / "spotify_data.csv"
    app.state.df_usuario = pd.read_csv(csv_path)
    logging.info("se ha obtenido el csv")
    return {"se ha obtenido el csv": f"{csv_path}"}

#emula el envio de datos me imagino que para el dashboard eso ya se tiene que ver
@app.get("/enviar_datos_emul")
async def enviar_datos_emul(x_api_key: str | None = Header(None, alias="x-api-key")):
    if app.state.df_usuario is None:
        logging.error("Datos no cargados, no se ha ejecutado la obtencion de datos")
        return {"error": "Datos no cargados. Ejecuta primero /obtener_datos_emul"}

    if app.state.df_usuario.empty:
        logging.error("No hay datos para enviar")
        return {"error": "No hay datos para enviar"}
    else:
        fila = app.state.df_usuario.iloc[0]
        fila_dict = fila.to_dict()
        return fila_dict

# Endpoint que con la entrada de datos predice una serie de 5 canciones del mismo cluster 
@app.post("/prediccion")
async def predecir_popularidad(cancion: CancionEntrada, x_api_key: str | None = Header(None, alias="x-api-key")):

    try:
        scaler_path = MODELOS_DIR / "scaler.pkl"
        modelo_path = MODELOS_DIR / "kmeans_spotify_model.pkl"
        csv_path = Path("datasets") / "canciones_clusterizadas.csv"
        scaler = joblib.load(scaler_path)
        modelo = joblib.load(modelo_path)
        clust_feats = ['instrumentalness', 'speechiness', 'danceability', 'valence', 'tempo']
        df_input = pd.DataFrame([cancion.model_dump()])[clust_feats]
        df_scaled = scaler.transform(df_input)
        cluster = int(modelo.predict(df_scaled)[0])
        df_canciones = pd.read_csv(csv_path)
        df_cluster = df_canciones[df_canciones["cluster"] == cluster]
        recomendaciones = df_cluster.sample(
            n=min(5, len(df_cluster)),
            random_state=42
        )
        resultado = recomendaciones[["track_name", "artist_name"]].to_dict(orient="records")
        # Devuelve el cluster con las canciones recomendaciones
        if cluster and resultado is not None:
            logging.info(f"Se ha generado una prediccion \n {resultado} \n correctamente para el cluster {cluster}")
        return {
            "modelo": "kmeans_spotify_model.pkl",
            "cluster_predicho": cluster,
            "recomendaciones": resultado
        }
    except Exception as e:
        logging.error(f"Error en el endpoint de prediccion: {e}")



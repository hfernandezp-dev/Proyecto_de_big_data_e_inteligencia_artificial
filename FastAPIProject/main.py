from fastapi import FastAPI, Header, HTTPException,Request,Depends,Security
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException as FastAPIHTTPException
from fastapi.security import APIKeyHeader
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
dataframes_dir = BASE_DIR / "src" / "datasets"
env_path = BASE_DIR / "environment" / ".env"
scaler_path = MODELOS_DIR / "scaler.pkl"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("SPOTIFY_API_KEY")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)




async def validar_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")

#simula la entrada de datos ya que no se ha desplegado
@app.get("/obtener_datos_emul",dependencies=[Depends(validar_api_key)])
async def obtener2_datos_emul():
    csv_path = BASE_DIR / "src" / "datasets" / "spotify_data.csv"
    app.state.df_usuario = pd.read_csv(csv_path)
    logging.info("se ha obtenido el csv")
    return {"se ha obtenido el csv": f"{csv_path}"}

#emula el envio de datos me imagino que para el dashboard eso ya se tiene que ver
@app.get("/enviar_datos_emul",dependencies=[Depends(validar_api_key)])
async def enviar_datos_emul():
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
@app.post("/prediccion"
    ,summary="Predice una serie de 5 canciones del mismo cluster"
    ,description="Recibe una canción como entrada y devuelve una lista de 5 canciones del mismo cluster"
    ,dependencies=[Depends(validar_api_key)])
async def predecir_popularidad(cancion: CancionEntrada):

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
        df_canciones = pd.read_csv(os.path.join(dataframes_dir, "canciones_clusterizadas.csv"))
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


@app.exception_handler(FastAPIHTTPException)
async def http_exception_handler(request: Request, exc: FastAPIHTTPException):
    logging.error(
        f"HTTP {exc.status_code} | "
        f"Path: {request.url.path} | "
        f"Detail: {exc.detail}"
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )
@app.get("/health")
def health():
    return {"status": "ok"}
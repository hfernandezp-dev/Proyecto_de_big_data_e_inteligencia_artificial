import subprocess
import sys

from fastapi import FastAPI, Header, HTTPException,Request,Depends,Security
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException as FastAPIHTTPException
from fastapi.security import APIKeyHeader
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
import pandas as pd
import requests
import os
import base64
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

origins = [
    "http://localhost:4200",
    "http://localhost:5500",
    "http://localhost:8080",
    "http://192.168.1.40:5500",
    "http://localhost:5500",
    "http://localhost:63342"
]


app = FastAPI(
    title="API de Recomendación Musical",
    description="API para recomendar canciones usando clustering KMeans",
    version="0.5.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELOS_DIR = BASE_DIR / "src" / "modelos"
dataframes_dir = BASE_DIR / "src" / "datasets"
env_path = BASE_DIR / "environment" / ".env"
scaler_path = MODELOS_DIR / "scaler.pkl"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("SPOTIFY_API_KEY")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

REDIRECT_URI = "https://127.0.0.1/callback"
CLIENT_ID = "9ac071fd07fd47768bcdbbc5f2fa3566"
CLIENT_SECRET = "c82569bd56d249ada2aa9d72ab0a8ee9"
access_tokens = {}

@app.get("/login")
def login():
    scope = "user-read-recently-played"
    url = (
        "https://accounts.spotify.com/authorize"
        f"?client_id={CLIENT_ID}"
        "&response_type=code"
        f"&redirect_uri={REDIRECT_URI}"
        f"&scope={scope}"
    )
    return RedirectResponse(url)

@app.get("/runflow")
async def run_flow():
    FLOW_PATH = BASE_DIR / "src" / "flows" / "spotify_flow.py"
    result = subprocess.run([sys.executable, str(FLOW_PATH)], capture_output=True, text=True)
    return {
        "status": "Flow ejecutado",
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }

@app.get("/callback")
def callback(code: str):
    token_url = "https://accounts.spotify.com/api/token"
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    r = requests.post(token_url, data=data)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    tokens = r.json()
    access_token = tokens["access_token"]
    refresh_token = tokens.get("refresh_token")

    # Guardamos el access_token temporalmente
    access_tokens["user"] = access_token

    return {"message": "Login exitoso", "access_token": access_token, "refresh_token": refresh_token}




def get_spotify_token():
    """
    Obtiene un token temporal de Spotify usando Client Credentials
    """
    auth_str = f"{CLIENT_ID}:{CLIENT_SECRET}"
    b64_auth_str = base64.b64encode(auth_str.encode()).decode()

    response = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={"Authorization": f"Basic {b64_auth_str}"},
        data={"grant_type": "client_credentials"}
    )
    return response.json()["access_token"]


async def validar_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")
@app.get("/api/recent-tracks")
def get_recent_tracks(access_token: str):
    url = "https://api.spotify.com/v1/me/player/recently-played?limit=50"
    headers = {"Authorization": f"Bearer {access_token}"}

    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        data = r.json()
        tracks = []
        for item in data["items"]:
            track = item["track"]
            artist_id = track["artists"][0]["id"] if track["artists"] else None
            tracks.append({
                "name": track["name"],
                "artists": [a["name"] for a in track["artists"]],
                "artist_id": artist_id,
                "album": track["album"]["name"],
                "played_at": item["played_at"],
                "image": track["album"]["images"][0]["url"] if track["album"]["images"] else None
            })
        return tracks
    else:
        raise HTTPException(status_code=r.status_code, detail=r.text)



@app.get("/api/generos")
async def get_generos(ids:str,access_token: str):
    artist_ids_list = ids.split(',')
    if(len(artist_ids_list)>50):
        raise HTTPException(status_code=400, detail="spotify solo permite 50 id")

    url = "https://api.spotify.com/v1/artists"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "ids": ids
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        detail = e.response.json().get("error", {"message": "Error desconocido de Spotify"}).get("message")
        raise HTTPException(status_code=status_code, detail=f"Error al obtener artistas de Spotify: {detail}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")

@app.get("/api/conseguir_canciones")
async def get_conseguir_canciones(ids:str,access_token: str):
    try:
        artist_ids_list = ids.split(',')
        if (len(artist_ids_list) > 50):
            raise HTTPException(status_code=400, detail="spotify solo permite 50 id")

        url = "https://api.spotify.com/v1/tracks"
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        params = {
            "ids": ids
        }
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        logging.info("se ha obtenido datos de las canciones pedidas")
        return response.json()
    except Exception as e:
        logging.error(f"error en la carga de ids: {e}")
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        logging.error(f"error al conectarse con la api con codigo: {status_code}")




#simula la entrada de datos ya que no se ha desplegado
@app.get("/obtener_datos_emul"
    ,summary="Simula la entrada de datos ya sea para un dashboard o un aplicacion"
    ,description="Esta ruta simula la entrada de datos ya sea para un dashboard o una aplicacion, para ello se necesita que se hayan obtenido los datos con la ruta /obtener_datos_emul"
    ,dependencies=[Depends(validar_api_key)])
async def obtener2_datos_emul():
    csv_path = BASE_DIR / "src" / "datasets" / "spotify_data.csv"
    app.state.df_usuario = pd.read_csv(csv_path)
    df_usuarioEmul=pd.read_csv(csv_path)
    fila = df_usuarioEmul.sample(n=1)
    fila_json = fila.to_dict(orient="records")[0]
    logging.info("se ha obtenido el csv")

    return fila_json
    # return {"se ha obtenido el csv": f"{csv_path}"}

#emula el envio de datos me imagino que para el dashboard eso ya se tiene que ver
@app.get("/enviar_datos_emul"
    ,summary="Simula el envio de datos ya sea para un dashboard o un aplicacion"
    ,description="Esta ruta simula el envio de datos ya sea para un dashboard o una aplicacion, para ello se necesita que se hayan obtenido los datos con la ruta /obtener_datos_emul"
    ,dependencies=[Depends(validar_api_key)])
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
        resultado = recomendaciones[["track_name", "artist_name","track_id"]].to_dict(orient="records")
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
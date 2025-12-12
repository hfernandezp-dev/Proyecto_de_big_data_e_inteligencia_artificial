from fastapi import FastAPI, Header, HTTPException, Depends
from pathlib import Path
import pandas as pd
import os
from dotenv import load_dotenv
import joblib
from schemas import CancionEntrada

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
MODELOS_DIR = BASE_DIR / "src" / "modelos"
env_path = BASE_DIR / "environment" / ".env"
load_dotenv(dotenv_path=env_path)

csv_path2 = BASE_DIR / "src" / "datasets" / "spotify_data.csv"

df_canciones2 = pd.read_csv(csv_path2)
modelo_path2 = MODELOS_DIR / "kmeans_spotify_model.pkl"
scaler_path = MODELOS_DIR / "scaler.pkl"
kmeans = joblib.load(modelo_path2)
scaler = joblib.load(scaler_path)
clust_feats = ['instrumentalness', 'speechiness', 'danceability', 'valence', 'tempo']

X = df_canciones2[clust_feats]
X_scaled = scaler.transform(X)
df_canciones2['cluster'] = kmeans.predict(X_scaled)
df_canciones2.to_csv("datasets/canciones_clusterizadas.csv", index=False)







API_KEY = os.getenv("SPOTIFY_API_KEY")

app.state.df_usuario = None
print("API_KEY:", API_KEY)
print("BASE_DIR:", BASE_DIR)


@app.get("/")
async def root(x_api_key: str | None = Header(None, alias="x-api-key")):

    modelos = [f.name for f in MODELOS_DIR.glob("*.pkl")]
    return {
        "message": "Hello World",
        "ruta_modelos": str(MODELOS_DIR),
        "modelos_encontrados": modelos
    }


@app.get("/hello/{name}")
async def say_hello(name: str,x_api_key: str | None = Header(None, alias="x-api-key")):
    return {"message": f"Hello {name}"}


#simula la entrada de datos ya que no se ha desplegado
@app.get("/obtener_datos_emul")
async def obtener2_datos_emul(x_api_key: str | None = Header(None, alias="x-api-key")):
    csv_path = BASE_DIR / "src" / "datasets" / "spotify_data.csv"
    app.state.df_usuario = pd.read_csv(csv_path)
    return {"se ha obtenido el csv": f"{csv_path}"}

#emula el envio de datos me imagino que para el dashboard eso ya se tiene que ver
@app.get("/enviar_datos_emul")
async def enviar_datos_emul(x_api_key: str | None = Header(None, alias="x-api-key")):
    if app.state.df_usuario is None:
        return {"error": "Datos no cargados. Ejecuta primero /obtener_datos_emul"}

    if app.state.df_usuario.empty:
        return {"message": "No hay datos para enviar"}
    else:
        fila = app.state.df_usuario.iloc[0]
        fila_dict = fila.to_dict()
        return fila_dict


@app.post("/prediccion")
async def predecir_popularidad(cancion: CancionEntrada, x_api_key: str | None = Header(None, alias="x-api-key")):
    # modelo_path = MODELOS_DIR / "kmeans_spotify_model.pkl"
    # modelo = joblib.load(modelo_path)
    # df_input = pd.DataFrame([cancion.model_dump()])
    # pred=modelo.predict(df_input)[0]

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
    #Devuelve el cluster con las canciones recomendaciones
    return {
        "modelo": "kmeans_spotify_model.pkl",
        "cluster_predicho": cluster,
        "recomendaciones":resultado
    }



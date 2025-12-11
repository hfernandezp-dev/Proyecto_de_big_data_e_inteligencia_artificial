from fastapi import FastAPI, Header, HTTPException, Depends
from pathlib import Path
import pandas as pd
import os
from dotenv import load_dotenv
app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
MODELOS_DIR = BASE_DIR / "src" / "modelos"
env_path = BASE_DIR / "environment" / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("SPOTIFY_API_KEY")

app.state.df_usuario = None
print("API_KEY:", API_KEY)
print("BASE_DIR:", BASE_DIR)

def check_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")


@app.get("/")
async def root(api_key: str = Depends(check_api_key)):

    modelos = [f.name for f in MODELOS_DIR.glob("*.pkl")]
    return {
        "message": "Hello World",
        "ruta_modelos": str(MODELOS_DIR),
        "modelos_encontrados": modelos
    }


@app.get("/hello/{name}")
async def say_hello(name: str,api_key: str = Depends(check_api_key)):
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
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from src.Entrenamiento import Cargar_Datos_Ruta, entrenamiento, iniciar_logger

scaler_global=None

def _crear_parquets_dummy(base_dir):
    """
    Crea ficheros X_train.parquet y X_test.parquet mínimos
    para que Cargar_Datos_Ruta() funcione.
    """
    datasets_dir = base_dir / "datasets"
    datasets_dir.mkdir()

    clust_feats = ['instrumentalness', 'speechiness', 'danceability', 'valence', 'tempo']
    cols_texto = ['artist_name', 'track_name', 'track_id', 'genre']

    df_train = pd.DataFrame({
        'instrumentalness': [0.1, 0.2, 0.3],
        'speechiness': [0.05, 0.1, 0.2],
        'danceability': [0.7, 0.8, 0.6],
        'valence': [0.4, 0.5, 0.6],
        'tempo': [120, 130, 140],
        'artist_name': ['a1', 'a2', 'a3'],
        'track_name': ['t1', 't2', 't3'],
        'track_id': ['id1', 'id2', 'id3'],
        'genre': ['rock', 'pop', 'metal'],
    })

    df_test = df_train.copy()

    df_train.to_parquet(datasets_dir / "X_train.parquet", index=False)
    df_test.to_parquet(datasets_dir / "X_test.parquet", index=False)


def test_cargar_datos_ruta_devuelve_arrays_escalados(tmp_path, monkeypatch):
    """
    Debe devolver dos arrays numpy con 5 columnas (las de clust_feats),
    sin las columnas de texto.
    """
    monkeypatch.chdir(tmp_path)
    _crear_parquets_dummy(tmp_path)

    iniciar_logger()
    X_train_scaled, X_test_scaled,scaler = Cargar_Datos_Ruta()
    scaler_global=scaler

    # Tipos
    assert isinstance(X_train_scaled, np.ndarray)
    assert isinstance(X_test_scaled, np.ndarray)

    assert X_train_scaled.shape == (3, 5)
    assert X_test_scaled.shape == (3, 5)


def test_entrenamiento_guarda_modelo(tmp_path, monkeypatch):
    """
    Debe entrenar un KMeans y guardar el modelo en la carpeta 'modelos'.
    """
    monkeypatch.chdir(tmp_path)
    iniciar_logger()

    X_dummy = np.random.rand(10, 5)

    entrenamiento(X_dummy,3,42,scaler_global)

    ruta_modelo = tmp_path / "modelos" / "kmeans_spotify_model.pkl"
    assert ruta_modelo.exists(), "No se ha guardado el modelo entrenado"
    
def test_cargar_datos_ruta_num_filas_correctas(tmp_path, monkeypatch):
    """
    Verifica que Cargar_Datos_Ruta devuelve el mismo número de filas
    que los parquets creados (en este caso, 3).
    """
    monkeypatch.chdir(tmp_path)
    _crear_parquets_dummy(Path(tmp_path))

    iniciar_logger()
    X_train_scaled, X_test_scaled,scaler = Cargar_Datos_Ruta()
    scaler_global = scaler

    assert X_train_scaled.shape[0] == 3
    assert X_test_scaled.shape[0] == 3
    
def test_entrenamiento_crea_directorio_modelos(tmp_path, monkeypatch):
    """
    Verifica que la función entrenamiento crea la carpeta 'modelos'
    al guardar el modelo entrenado.
    """
    monkeypatch.chdir(tmp_path)
    iniciar_logger()

    X_dummy = np.random.rand(5, 5)
    entrenamiento(X_dummy,3,42,scaler_global)

    modelos_dir = Path(tmp_path) / "modelos"
    assert modelos_dir.is_dir(), "No se ha creado el directorio 'modelos'"

def test_cargar_datos_ruta_estandariza_features(tmp_path, monkeypatch):
    """
    Comprueba que los datos de entrenamiento quedan estandarizados:
    media ~ 0 y desviación típica ~ 1 en cada columna.
    """
    monkeypatch.chdir(tmp_path)
    _crear_parquets_dummy(Path(tmp_path))

    iniciar_logger()
    X_train_scaled, X_test_scaled,scaler = Cargar_Datos_Ruta()

    medias = X_train_scaled.mean(axis=0)
    desvios = X_train_scaled.std(axis=0)

    assert np.allclose(medias, 0, atol=1e-6)
    assert np.allclose(desvios, 1, atol=1e-6)


def test_modelo_guardado_es_kmeans(tmp_path, monkeypatch):
    """
    Comprueba que el modelo guardado es un KMeans entrenado
    y que tiene atributo cluster_centers_ y n_clusters > 1.
    """
    monkeypatch.chdir(tmp_path)
    iniciar_logger()

    X_dummy = np.random.rand(20, 5)
    entrenamiento(X_dummy,2,42,scaler_global)

    ruta_modelo = Path(tmp_path) / "modelos" / "kmeans_spotify_model.pkl"
    modelo = joblib.load(ruta_modelo)

    assert hasattr(modelo, "cluster_centers_")
    assert hasattr(modelo, "n_clusters")
    assert modelo.n_clusters > 1


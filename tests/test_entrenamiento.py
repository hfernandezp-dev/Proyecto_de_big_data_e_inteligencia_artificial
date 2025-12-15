import logging
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import joblib
import pytest

from src.Entrenamiento import Cargar_Datos_Ruta, entrenamiento, iniciar_logger


# ============================================================
# Helpers
# ============================================================
def _crear_parquets_dummy(base_dir: Path, n_rows: int = 10):
    """
    Crea datasets/X_train.parquet y datasets/X_test.parquet con datos mínimos.
    Usamos n_rows >= 4 para que el __main__ (K=4) no falle.
    """
    datasets_dir = base_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # columnas numéricas (features)
    rng = np.random.default_rng(123)
    df = pd.DataFrame({
        "instrumentalness": rng.random(n_rows),
        "speechiness": rng.random(n_rows),
        "danceability": rng.random(n_rows),
        "valence": rng.random(n_rows),
        "tempo": rng.integers(80, 180, size=n_rows),
        # columnas texto que Entrenamiento.py elimina
        "artist_name": [f"a{i}" for i in range(n_rows)],
        "track_name": [f"t{i}" for i in range(n_rows)],
        "track_id": [f"id{i}" for i in range(n_rows)],
        "genre": ["rock"] * n_rows,
    })

    df.to_parquet(datasets_dir / "X_train.parquet", index=False)
    df.to_parquet(datasets_dir / "X_test.parquet", index=False)


@pytest.fixture
def datos_y_scaler(tmp_path, monkeypatch):
    """
    Fixture que crea parquets y devuelve (X_train_scaled, scaler).
    """
    monkeypatch.chdir(tmp_path)
    iniciar_logger()
    _crear_parquets_dummy(Path(tmp_path), n_rows=10)

    X_train_scaled, X_test_scaled, scaler = Cargar_Datos_Ruta()
    return X_train_scaled, scaler


# ============================================================
# Tests Cargar_Datos_Ruta()
# ============================================================
def test_cargar_datos_ruta_devuelve_arrays_escalados(tmp_path, monkeypatch):
    """
    Debe devolver dos arrays numpy con 5 columnas, sin columnas de texto.
    """
    monkeypatch.chdir(tmp_path)
    iniciar_logger()
    _crear_parquets_dummy(Path(tmp_path), n_rows=10)

    X_train_scaled, X_test_scaled, scaler = Cargar_Datos_Ruta()

    assert isinstance(X_train_scaled, np.ndarray)
    assert isinstance(X_test_scaled, np.ndarray)
    assert X_train_scaled.shape == (10, 5)
    assert X_test_scaled.shape == (10, 5)
    assert scaler is not None


def test_cargar_datos_ruta_estandariza_features(tmp_path, monkeypatch):
    """
    Media ~ 0 y std ~ 1 en cada feature del train (StandardScaler).
    """
    monkeypatch.chdir(tmp_path)
    iniciar_logger()
    _crear_parquets_dummy(Path(tmp_path), n_rows=10)

    X_train_scaled, _, _ = Cargar_Datos_Ruta()

    medias = X_train_scaled.mean(axis=0)
    desvios = X_train_scaled.std(axis=0)

    assert np.allclose(medias, 0, atol=1e-6)
    assert np.allclose(desvios, 1, atol=1e-6)


def test_cargar_datos_ruta_si_fallan_parquets_loguea_error(tmp_path, monkeypatch, caplog):
    """
    Cubre el except de Cargar_Datos_Ruta(): si no existen los parquets debe loguear error y devolver None.
    """
    monkeypatch.chdir(tmp_path)
    iniciar_logger()

    with caplog.at_level(logging.ERROR):
        out = Cargar_Datos_Ruta()

    assert out is None
    assert any("Error en la carga de datos" in r.getMessage() for r in caplog.records)


# ============================================================
# Tests entrenamiento()
# ============================================================
def test_entrenamiento_guarda_modelo_y_scaler(tmp_path, monkeypatch, datos_y_scaler):
    """
    Debe guardar kmeans_spotify_model.pkl y scaler.pkl dentro de modelos/.
    """
    monkeypatch.chdir(tmp_path)
    iniciar_logger()

    X_train_scaled, scaler = datos_y_scaler

    res = entrenamiento(X_train_scaled, k=3, random_state=42, scaler=scaler)
    assert isinstance(res, dict)
    assert "model" in res

    ruta_modelo = Path(tmp_path) / "modelos" / "kmeans_spotify_model.pkl"
    ruta_scaler = Path(tmp_path) / "modelos" / "scaler.pkl"

    assert ruta_modelo.exists()
    assert ruta_scaler.exists()

    modelo = joblib.load(ruta_modelo)
    scaler_guardado = joblib.load(ruta_scaler)

    assert hasattr(modelo, "cluster_centers_")
    # el scaler guardado debe existir y ser "algo" compatible
    assert scaler_guardado is not None


def test_entrenamiento_k_eq_1_silhouette_es_0(tmp_path, monkeypatch, datos_y_scaler):
    """
    Cubre la rama: silhouette_score(...) if k > 1 else 0
    """
    monkeypatch.chdir(tmp_path)
    iniciar_logger()

    X_train_scaled, scaler = datos_y_scaler

    res = entrenamiento(X_train_scaled, k=1, random_state=42, scaler=scaler)

    assert res["k"] == 1
    assert res["silhouette"] == 0
    assert res["wcss"] >= 0
    assert res["duration_sec"] >= 0


def test_entrenamiento_error_loguea(tmp_path, monkeypatch, caplog):
    """
    Fuerza un error en entrenamiento (x_train inválido) para cubrir el except.
    """
    monkeypatch.chdir(tmp_path)
    iniciar_logger()

    with caplog.at_level(logging.ERROR):
        out = entrenamiento(None, k=2, random_state=42, scaler=None)

    assert out is None
    assert any("Error en entrenamiento" in r.getMessage() for r in caplog.records)


# ============================================================
# Cubre el bloque if __name__ == "__main__"
# ============================================================
def test_ejecutar_modulo_como_main_cubre_main(tmp_path, monkeypatch):
    """
    Ejecuta src.Entrenamiento como script para cubrir el __main__.
    OJO: necesita >=4 filas porque el __main__ usa K=4.
    """
    monkeypatch.chdir(tmp_path)
    _crear_parquets_dummy(Path(tmp_path), n_rows=10)

    # Ejecuta el módulo como si fuese `python -m src.Entrenamiento`
    runpy.run_module("src.Entrenamiento", run_name="__main__")

    # Si ha corrido, debería haber intentado crear logs y/o modelos
    assert (Path(tmp_path) / "logs").exists()

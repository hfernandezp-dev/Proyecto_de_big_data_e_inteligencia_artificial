from pathlib import Path
import pandas as pd
import pytest
import logging

from src.Pipeline import (
    iniciar_logger,
    iniciar_spark,
    dividir_dataset,
    # limpieza_datos,  # seguimos sin llamarla para no arrancar Spark real
)


@pytest.mark.skip(reason="Se omite este test para no arrancar Spark en los tests unitarios.")
def test_limpieza_datos_filtra_generos_y_outliers(tmp_path, monkeypatch):
    """
    Test desactivado: limpieza_datos lanza Spark y es demasiado pesado
    para los tests unitarios en este entorno.
    """
    pass


def test_iniciar_spark_devuelve_fake_sesion(monkeypatch):
    """
    Comprueba que iniciar_spark devuelve un 'spark' con atributo version,
    simulando SparkSession para no levantar Spark real.
    """
    # Evitar que findspark haga nada real
    monkeypatch.setattr("src.Pipeline.findspark.init", lambda: None, raising=False)

    # Fake Spark y builder para evitar getOrCreate real
    class FakeSpark:
        def __init__(self):
            self.version = "3.5.0"

    class FakeBuilder:
        def master(self, *args, **kwargs):
            return self

        def appName(self, *args, **kwargs):
            return self

        def getOrCreate(self):
            return FakeSpark()

    # Parcheamos pyspark.sql.SparkSession para que use nuestro builder falso
    import pyspark.sql
    monkeypatch.setattr(pyspark.sql, "SparkSession", type("FakeSparkSession", (), {"builder": FakeBuilder()}))

    iniciar_logger()
    spark = iniciar_spark()

    assert spark is not None
    assert hasattr(spark, "version")
    assert spark.version == "3.5.0"


def test_dividir_dataset_crea_parquets(tmp_path, monkeypatch):
    """
    dividir_dataset debe crear los 4 ficheros parquet en la carpeta datasets/.
    """
    monkeypatch.chdir(tmp_path)
    iniciar_logger()

    # Crear la carpeta datasets/ porque la función escribe rutas relativas
    datasets_dir = Path(tmp_path) / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # DataFrame pequeñito con la columna 'popularity'
    df = pd.DataFrame({
        "instrumentalness": [0.1, 0.2, 0.3, 0.4],
        "speechiness": [0.05, 0.1, 0.15, 0.2],
        "danceability": [0.6, 0.7, 0.8, 0.9],
        "valence": [0.3, 0.4, 0.5, 0.6],
        "tempo": [110, 120, 130, 140],
        "popularity": [40, 50, 60, 70],
    })

    dividir_dataset(df)

    assert (datasets_dir / "X_train.parquet").exists()
    assert (datasets_dir / "X_test.parquet").exists()
    assert (datasets_dir / "y_train.parquet").exists()
    assert (datasets_dir / "y_test.parquet").exists()


def test_dividir_dataset_respeta_columnas(tmp_path, monkeypatch):
    """
    Comprueba que las columnas de X no incluyen 'popularity'
    y que y solo contiene esa columna.
    """
    monkeypatch.chdir(tmp_path)
    iniciar_logger()

    datasets_dir = Path(tmp_path) / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "instrumentalness": [0.1, 0.2, 0.3, 0.4],
        "speechiness": [0.05, 0.1, 0.15, 0.2],
        "danceability": [0.6, 0.7, 0.8, 0.9],
        "valence": [0.3, 0.4, 0.5, 0.6],
        "tempo": [110, 120, 130, 140],
        "popularity": [40, 50, 60, 70],
    })

    dividir_dataset(df)

    X_train = pd.read_parquet(datasets_dir / "X_train.parquet")
    y_train = pd.read_parquet(datasets_dir / "y_train.parquet")

    # En X no debe estar 'popularity'
    assert "popularity" not in X_train.columns

    # En y solo debe estar 'popularity'
    assert list(y_train.columns) == ["popularity"]


def test_iniciar_logger_no_falla_en_directorio_tmp(tmp_path, monkeypatch):
    """
    Comprobar que iniciar_logger se puede ejecutar sin errores
    en un directorio temporal vacío.
    """
    monkeypatch.chdir(tmp_path)
    iniciar_logger()
    # Si no lanza excepción, el test se considera pasado.


def test_dividir_dataset_no_modifica_dataframe_original(tmp_path, monkeypatch):
    """
    Verifica que dividir_dataset no cambia el DataFrame original
    (mismas filas y mismas columnas).
    """
    monkeypatch.chdir(tmp_path)
    iniciar_logger()

    datasets_dir = Path(tmp_path) / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "instrumentalness": [0.1, 0.2, 0.3, 0.4],
        "speechiness": [0.05, 0.1, 0.15, 0.2],
        "danceability": [0.6, 0.7, 0.8, 0.9],
        "valence": [0.3, 0.4, 0.5, 0.6],
        "tempo": [110, 120, 130, 140],
        "popularity": [40, 50, 60, 70],
    })

    df_original = df.copy(deep=True)

    dividir_dataset(df)

    # Comprobamos que df no ha cambiado
    assert df.equals(df_original)


def test_dividir_dataset_maneja_errores_sin_explotar(tmp_path, monkeypatch, caplog):
    """
    Fuerza un error en la escritura (sin carpeta datasets) y comprueba
    que la función no lanza excepción y registra un log de error.
    """
    monkeypatch.chdir(tmp_path)
    iniciar_logger()

    # NO creamos datasets_dir para provocar el error
    df = pd.DataFrame({
        "instrumentalness": [0.1, 0.2],
        "speechiness": [0.05, 0.1],
        "danceability": [0.6, 0.7],
        "valence": [0.3, 0.4],
        "tempo": [110, 120],
        "popularity": [40, 50],
    })

    with caplog.at_level(logging.ERROR):
        dividir_dataset(df)

    # No debe lanzar excepción y debe haberse logado un error
    assert any("Error en la dividir de datos" in rec.getMessage() for rec in caplog.records)

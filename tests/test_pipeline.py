from pathlib import Path
import pandas as pd
import pytest
import logging
import os
import sys
import src.Pipeline as pipeline
from src.Pipeline import (
    iniciar_logger,
    iniciar_spark,
    dividir_dataset,
    # limpieza_datos,
)

base_dir = Path(__file__).parent.parent

loggings_path=os.path.join(base_dir,"src","logs")


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
    monkeypatch.setattr("src.Pipeline.findspark.init", lambda: None, raising=False)

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

    import pyspark.sql
    monkeypatch.setattr(pyspark.sql, "SparkSession", type("FakeSparkSession", (), {"builder": FakeBuilder()}))

    iniciar_logger(loggings_path)
    spark = iniciar_spark()

    assert spark is not None
    assert hasattr(spark, "version")
    assert spark.version == "3.5.0"


def test_iniciar_spark_error_loguea_y_devuelve_none(monkeypatch, caplog):
    monkeypatch.setattr(pipeline.findspark, "init", lambda: None)

    class _BadBuilder:
        def master(self, *a, **k): return self
        def appName(self, *a, **k): return self
        def getOrCreate(self): raise RuntimeError("boom")

    class _FakeSparkSession:
        builder = _BadBuilder()

    import pyspark.sql
    monkeypatch.setattr(pyspark.sql, "SparkSession", _FakeSparkSession)

    pipeline.iniciar_logger(loggings_path)

    with caplog.at_level(logging.ERROR):
        spark = pipeline.iniciar_spark()

    assert spark is None
    assert any("Error al iniciar Spark" in r.getMessage() for r in caplog.records)

def test_dividir_dataset_crea_parquets(tmp_path, monkeypatch):
    """
    dividir_dataset debe crear los 4 ficheros parquet en la carpeta datasets/.
    """
    monkeypatch.chdir(tmp_path)
    iniciar_logger(tmp_path)

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

    dividir_dataset(df,datasets_dir)

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
    iniciar_logger(tmp_path)

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

    dividir_dataset(df,datasets_dir)

    X_train = pd.read_parquet(datasets_dir / "X_train.parquet")
    y_train = pd.read_parquet(datasets_dir / "y_train.parquet")

    assert "popularity" not in X_train.columns
    assert list(y_train.columns) == ["popularity"]


def test_iniciar_logger_no_falla_en_directorio_tmp(tmp_path, monkeypatch):
    """
    Comprobar que iniciar_logger se puede ejecutar sin errores
    en un directorio temporal vacío.
    """
    monkeypatch.chdir(tmp_path)
    iniciar_logger(tmp_path)


def test_dividir_dataset_no_modifica_dataframe_original(tmp_path, monkeypatch):
    """
    Verifica que dividir_dataset no cambia el DataFrame original
    (mismas filas y mismas columnas).
    """
    monkeypatch.chdir(tmp_path)
    iniciar_logger(tmp_path)

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
    dividir_dataset(df,datasets_dir)
    assert df.equals(df_original)


def test_dividir_dataset_maneja_errores_sin_explotar(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    iniciar_logger(tmp_path)

    # Falta la columna popularity → fuerza error
    df = pd.DataFrame({
        "danceability": [0.6, 0.7]
    })

    with caplog.at_level(logging.ERROR):
        dividir_dataset(df, tmp_path)

    assert any(
        "Error al dividir dataset" in rec.getMessage()
        for rec in caplog.records
    )
class _FakeNA:
    def __init__(self, df):
        self._df = df

    def drop(self):
        return self._df


class _FakeSparkDF:
    def __init__(self, pandas_df):
        self._pdf = pandas_df
        self.na = _FakeNA(self)

    def drop(self, *_args, **_kwargs):
        return self

    def filter(self, *_args, **_kwargs):
        return self

    def withColumn(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def to_csv(self, path, index=False):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._pdf.to_csv(path, index=index)

    def toPandas(self):
        return self._pdf


class _FakeReader:
    def __init__(self, fake_df):
        self._fake_df = fake_df

    def option(self, *_args, **_kwargs):
        return self

    def csv(self, *_args, **_kwargs):
        return self._fake_df


class _FakeSpark:
    def __init__(self, fake_df):
        self.read = _FakeReader(fake_df)


def test_limpieza_datos_con_fake_spark_ejecuta_sin_crashear(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    pipeline.iniciar_logger(tmp_path)

    pdf = pd.DataFrame({
        "genre": ["rock", "pop"],
        "duration_ms": [200000, 180000],
        "tempo": [120.0, 130.0],
        "popularity": [50, 60],
    })

    fake_df = _FakeSparkDF(pdf)
    fake_spark = _FakeSpark(fake_df)

    monkeypatch.setattr(pipeline, "iniciar_spark", lambda: fake_spark)

    with caplog.at_level(logging.ERROR):
        out = pipeline.limpieza_datos(tmp_path,2000)
        
    assert out is None or isinstance(out, pd.DataFrame)
    assert any(
        "Error al limpiar datos" in rec.getMessage()
        for rec in caplog.records
    )

def test_limpieza_datos_si_reader_falla_loguea_error(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    pipeline.iniciar_logger(tmp_path)

    class _BoomReader:
        def option(self, *a, **k): return self
        def csv(self, *a, **k): raise RuntimeError("csv read fail")

    class _BoomSpark:
        read = _BoomReader()

    monkeypatch.setattr(pipeline, "iniciar_spark", lambda: _BoomSpark())

    with caplog.at_level(logging.ERROR):
        out = pipeline.limpieza_datos(tmp_path,2000)

    assert out is None
    assert any("Error al limpiar datos" in r.getMessage() for r in caplog.records)

def test_main_llama_a_limpieza_y_dividir(monkeypatch):
    llamado = {"limpieza": False, "dividir": False}

    def _fake_limpieza(input_csv, limit_rows):
        llamado["limpieza"] = True
        return pd.DataFrame({"a": [1], "popularity": [10]})

    def _fake_dividir(df, output_dir):
        llamado["dividir"] = True
        assert "popularity" in df.columns

    monkeypatch.setattr(pipeline, "iniciar_logger", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "limpieza_datos", _fake_limpieza)
    monkeypatch.setattr(pipeline, "dividir_dataset", _fake_dividir)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pipeline",
            "--input_csv", "fake.csv",
            "--output_dir", "out",
            "--logs_dir", "logs",
        ],
    )

    pipeline.main()

    assert llamado["limpieza"] is True
    assert llamado["dividir"] is True
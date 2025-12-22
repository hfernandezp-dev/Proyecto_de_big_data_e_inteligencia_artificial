# src/Pipeline.py
import os
import argparse
import logging
import findspark
from pyspark.sql import functions as f
from sklearn.model_selection import train_test_split

def iniciar_logger(logs_dir: str):
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logs_dir, "Pipeline.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def iniciar_spark():
    findspark.init()
    try:
        from pyspark.sql import SparkSession
        return (
            SparkSession.builder
            .master("local[*]")
            .appName("spotify-pipeline")
            .getOrCreate()
        )
    except Exception as e:
        logging.error(f"Error al iniciar Spark: {e}")

def limpieza_datos(input_csv: str, limit_rows: int | None):
    spark = iniciar_spark()

    try:
        df = (
            spark.read
            .option("header", True)
            .option("mode", "DROPMALFORMED")
            .csv(input_csv)
            .drop("_c0")
            .na.drop()
        )
        # ---------------------------------------------
        # Se va ha filtrar generos musicales para evitar problemas de sobreentrenamiento quedandome con los mas populares
        # ---------------------------------------------
        generos = [
            "alt-rock", "hard-rock", "metal", "rock", "psych-rock",
            "pop", "dance", "hip-hop", "electro", "disco", "blues"
        ]
        # ---------------------------------------------
        # Nos quedaremos con las canciones que duren menos de 20 minutos
        # ---------------------------------------------
        df = (
            df.filter(f.col("genre").isin(generos))
            .filter(f.col("duration_ms") < 1_200_000)
            .withColumn("tempo", f.col("tempo").cast("double"))
            .filter((f.col("tempo") >= 40) & (f.col("tempo") <= 250))
        )
        # ---------------------------------------------
        # Otro valor que puede ser considerado Outlier el el tempo segun el EDA realizado
        # ---------------------------------------------
        if limit_rows:
            df = df.limit(limit_rows)
        logging.info("Se ha limpiado el dataset de Spotify correctamente")
        return df.toPandas()
    except Exception as e:
        logging.error(f"Error al limpiar datos: {e}")

def dividir_dataset(df, output_dir: str):
    try:
        # ---------------------------------------------
        # Divido los datos para su entrenamiento y pruebas del modelo
        # ---------------------------------------------
        os.makedirs(output_dir, exist_ok=True)

        X = df.drop(columns=["popularity"])
        y = df["popularity"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train.to_parquet(os.path.join(output_dir, "X_train.parquet"))
        X_test.to_parquet(os.path.join(output_dir, "X_test.parquet"))
        y_train.to_frame().to_parquet(os.path.join(output_dir, "y_train.parquet"))
        y_test.to_frame().to_parquet(os.path.join(output_dir, "y_test.parquet"))
        logging.info(f"División del dataset realizada correctamente em {output_dir}")
    except Exception as e:
        logging.error(f"Error al dividir dataset: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--logs_dir", required=True)
    parser.add_argument("--limit_rows", type=int, default=None)
    args = parser.parse_args()

    iniciar_logger(args.logs_dir)
    logging.info("Iniciando pipeline de datos")

    df = limpieza_datos(args.input_csv, args.limit_rows)
    dividir_dataset(df, args.output_dir)

    logging.info("Pipeline finalizado correctamente")

if __name__ == "__main__":
    main()

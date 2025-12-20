import logging
import os
import findspark
import argparse
from sklearn.model_selection import train_test_split
from pathlib import Path
from pyspark.sql import functions as f



def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline")

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=os.getenv("SPOTIFY_ARTIFACTS_DIR", Path(__file__).resolve().parent)
    )

    parser.add_argument(
        "--input-csv",
        type=Path,
        default=os.getenv("SPOTIFY_RAW_CSV", "datasets/spotify_data.csv")
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=os.getenv("SPOTIFY_RAW_CSV", "datasets/spotify_data.csv")
    )

    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=2000)

    return parser.parse_args()


# =========================================================
# LOGGER
# =========================================================

def iniciar_logger(logs_dir: Path):
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=logs_dir / "Pipeline.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        encoding="utf-8"
    )




# def iniciar_logger():
#
#     log_dir = "logs"
#     os.makedirs(log_dir, exist_ok=True)
#     logging.basicConfig(
#         filename=os.path.join(log_dir, "Pipeline.log"),
#         level=logging.INFO,
#         format="%(asctime)s - %(levelname)s - %(message)s"
#     )


def iniciar_spark():
    findspark.init()
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder \
            .master("local[*]") \
            .appName("ejercicio3") \
            .getOrCreate()
        logging.info(f"¡Clúster de Spark virtual iniciado con éxito! con la version {spark.version}")
        return spark
    except Exception as e:
        logging.error(f"Error al iniciar Spark: {e}")
        print("❌ No se pudo iniciar el clúster de Spark virtual. Revisa los logs para más detalles.")


def limpieza_datos(input_csv:Path,output:Path, sample_size:int):
   spark=iniciar_spark()
   try:
       df_spotify = (spark.read
                     .option("header",True)
                     .option("mode","DROPMALFORMED")
                     .option("multiLine", False)
                     .csv(str(input_csv), header=True))
       df_spotify = df_spotify.drop('_c0')
       df_spotify=df_spotify.na.drop()
       #---------------------------------------------
       #Se va ha filtrar generos musicales para evitar problemas de sobreentrenamiento quedandome con los mas populares
       #---------------------------------------------
       generos_para_mantener = ['alt-rock', 'hard-rock', 'metal', 'rock', 'psych-rock',
                      'pop', 'dance', 'hip-hop', 'electro', 'disco', 'blues']
       df_spotify=df_spotify.filter(f.col('genre').isin(generos_para_mantener))
       #---------------------------------------------
       #Nos quedaremos con las canciones que duren menos de 20 minutos
       #---------------------------------------------
       df_spotify=df_spotify.filter(f.col('duration_ms')<1200000)
       #---------------------------------------------
       #Otro valor que puede ser considerado Outlier el el tempo segun el EDA realizado
       #---------------------------------------------
       df_spotify = df_spotify.withColumn('tempo', f.col('tempo').cast("Double"))
       df_spotify = df_spotify.filter((f.col('tempo') >= 40) & (f.col('tempo') <= 250))
       logging.info("Se ha limpiado el dataset de Spotify correctamente")
       output.parent.mkdir(parents=True, exist_ok=True)
       df_spotify=df_spotify.limit(sample_size)
       df_pandas = df_spotify.toPandas()
       ruta_csv = os.path.join(output, "datasets", "spotify_data.csv")
       df_pandas.to_csv(ruta_csv, index=False)
       return df_pandas

   except Exception as e:
       logging.error(f"Error en la limpieza de datos:\n {e}")
def dividir_dataset(df,datasets_dir:Path, test_size,random_state):
    try:
        datasets_dir.mkdir(parents=True, exist_ok=True)
        # ---------------------------------------------
        # Divido los datos para su entrenamiento y pruebas del modelo
        # ---------------------------------------------

        X = df.drop(columns=['popularity'])
        Y = df['popularity']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
        X_train.to_parquet(os.path.join(datasets_dir, 'X_train.parquet'), index=False)
        X_test.to_parquet(os.path.join(datasets_dir, 'X_test.parquet'), index=False)
        y_train.to_frame().to_parquet(os.path.join(datasets_dir, 'y_train.parquet'), index=False)
        y_test.to_frame().to_parquet(os.path.join(datasets_dir, 'y_test.parquet'), index=False)
        logging.info(f"División del dataset realizada correctamente en {datasets_dir}")
    except Exception as e:
        logging.error(f"Error en la dividir de datos: {e}")


def main():
    args = parse_args()
    BASE_DIR = args.base_dir
    DATASETS_DIR = BASE_DIR / "datasets"
    LOGS_DIR = BASE_DIR / "logs"

    iniciar_logger(LOGS_DIR)
    df=limpieza_datos(args.input_csv,args.output_csv,args.sample_size)
    dividir_dataset(df,DATASETS_DIR,args.test_size,args.random_state)

if __name__ == "__main__":
    main()

# Ejemplo de uso
# logging.info("La aplicación ha iniciado")
# logging.warning("Esto es una advertencia")
# logging.error("Esto es un error")


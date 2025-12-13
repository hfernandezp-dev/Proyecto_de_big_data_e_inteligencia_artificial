import logging
import os
import findspark
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pyspark.sql import functions as f



def iniciar_logger():

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "Pipeline.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


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


def limpieza_datos():
   spark=iniciar_spark()
   try:
       df_spotify = (spark.read
                     .option("header",True)
                     .option("mode","DROPMALFORMED")
                     .option("multiLine", False)
                     .csv('datasets/spotify_data.csv', header=True))
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
       df_spotify=df_spotify.limit(2000)
       df_spotify.to_csv("datasets/spotify_data.csv", index=False)
       return df_spotify.toPandas()

   except Exception as e:
       logging.error(f"Error en la limpieza de datos:\n {e}")
def dividir_dataset(df):
    try:
        # ---------------------------------------------
        # Divido los datos para su entrenamiento y pruebas del modelo
        # ---------------------------------------------

        X = df.drop(columns=['popularity'])
        Y = df['popularity']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        RUTA_SALIDA = 'datasets/'
        X_train.to_parquet(os.path.join(RUTA_SALIDA, 'X_train.parquet'), index=False)
        X_test.to_parquet(os.path.join(RUTA_SALIDA, 'X_test.parquet'), index=False)
        y_train.to_frame().to_parquet(os.path.join(RUTA_SALIDA, 'y_train.parquet'), index=False)
        y_test.to_frame().to_parquet(os.path.join(RUTA_SALIDA, 'y_test.parquet'), index=False)
        logging.info(f"División del dataset realizada correctamente em {RUTA_SALIDA}")
    except Exception as e:
        logging.error(f"Error en la dividir de datos: {e}")


def main():
    iniciar_logger()
    df=limpieza_datos()
    dividir_dataset(df)

if __name__ == "__main__":
    main()

# Ejemplo de uso
# logging.info("La aplicación ha iniciado")
# logging.warning("Esto es una advertencia")
# logging.error("Esto es un error")


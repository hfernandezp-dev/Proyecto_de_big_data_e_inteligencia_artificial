import logging
import os
import findspark




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

    except Exception as e:
        logging.error(f"Error al iniciar Spark: {e}")
        print("❌ No se pudo iniciar el clúster de Spark virtual. Revisa los logs para más detalles.")


def limpieza_datos():






def main():
    iniciar_logger()
    iniciar_spark()


if __name__ == "__main__":
    main()

# Ejemplo de uso
# logging.info("La aplicación ha iniciado")
# logging.warning("Esto es una advertencia")
# logging.error("Esto es un error")


import logging
import os
import findspark

# Crear la carpeta "logs" si no existe
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Configurar logging
logging.basicConfig(
    filename=os.path.join(log_dir, "app.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Ejemplo de uso
logging.info("La aplicación ha iniciado")
logging.warning("Esto es una advertencia")
logging.error("Esto es un error")


try:
    findspark.init()

    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("ejercicio3") \
        .getOrCreate()
    print("✅ ¡Clúster de Spark virtual iniciado con éxito!")

    print(f"Versión de Spark: {spark.version}")

except Exception as e:
    logging.error(f"Error al iniciar Spark: {e}")
    print("❌ No se pudo iniciar el clúster de Spark virtual. Revisa los logs para más detalles.")


import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =========================================================
# CONFIGURACIÓN DE PERSISTENCIA (FUERA DE PREFECT TEMP)
# =========================================================

PERSISTENT_BASE_DIR = Path(
    os.getenv("SPOTIFY_ARTIFACTS_DIR", "C:/prefect_artifacts/spotify")
)

DATASETS_DIR = PERSISTENT_BASE_DIR / "datasets"
MODELOS_DIR = PERSISTENT_BASE_DIR / "modelos"
LOGS_DIR = PERSISTENT_BASE_DIR / "logs"

MODELOS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOGGER
# =========================================================

def iniciar_logger():
    logging.basicConfig(
        filename=LOGS_DIR / "Entrenamiento.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        encoding="utf-8"
    )


# =========================================================
# CARGA Y PREPROCESADO DE DATOS
# =========================================================

def Cargar_Datos_Ruta():
    try:
        logging.info("Cargando datasets desde ruta persistente")

        X_train = pd.read_parquet(DATASETS_DIR / "X_train.parquet")
        X_test = pd.read_parquet(DATASETS_DIR / "X_test.parquet")

        COLS_TEXTO = ["artist_name", "track_name", "track_id", "genre"]
        cols_a_eliminar = [c for c in COLS_TEXTO if c in X_train.columns]

        X_train.drop(columns=cols_a_eliminar, inplace=True)
        X_test.drop(columns=cols_a_eliminar, inplace=True)

        clust_feats = [
            "instrumentalness",
            "speechiness",
            "danceability",
            "valence",
            "tempo"
        ]

        X_train = X_train[clust_feats]
        X_test = X_test[clust_feats]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        logging.info("Datos cargados y escalados correctamente")

        return X_train_scaled, X_test_scaled, scaler

    except Exception:
        logging.exception("Error en la carga y preprocesado de datos")
        raise


# =========================================================
# ENTRENAMIENTO
# =========================================================

def entrenamiento(x_train, k, random_state, scaler):
    try:
        logging.info(f"Iniciando entrenamiento KMeans (k={k})")
        inicio = time.time()

        kmeans = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10
        )

        kmeans.fit(x_train)

        joblib.dump(kmeans, MODELOS_DIR / "kmeans_spotify_model.pkl")
        joblib.dump(scaler, MODELOS_DIR / "scaler.pkl")

        duracion = time.time() - inicio
        wcss = kmeans.inertia_
        silhouette = silhouette_score(x_train, kmeans.labels_) if k > 1 else 0

        logging.info(
            f"Entrenamiento finalizado | "
            f"WCSS={wcss:.2f} | Silhouette={silhouette:.4f} | "
            f"Duración={duracion:.2f}s"
        )

        return {
            "model": kmeans,
            "k": k,
            "wcss": wcss,
            "silhouette": silhouette,
            "duration_sec": duracion
        }

    except Exception:
        logging.exception("Error durante el entrenamiento")
        raise


# =========================================================
# EJECUCIÓN DIRECTA (TEST LOCAL / PREFECT)
# =========================================================

if __name__ == "__main__":
    iniciar_logger()

    logging.info("===== INICIO ENTRENAMIENTO SPOTIFY =====")

    x_train, x_test, scaler = Cargar_Datos_Ruta()

    resultados = entrenamiento(
        x_train=x_train,
        k=4,
        random_state=42,
        scaler=scaler
    )

    logging.info(
        f"Resultado final | "
        f"WCSS={resultados['wcss']:.2f} | "
        f"Silhouette={resultados['silhouette']:.4f}"
    )

    logging.info("===== FIN ENTRENAMIENTO SPOTIFY =====")

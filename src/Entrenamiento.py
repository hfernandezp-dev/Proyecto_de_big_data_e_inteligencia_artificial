import logging
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import joblib

def iniciar_logger():

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "Entrenamiento.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def Cargar_Datos_Ruta():
    try:
        X_train = pd.read_parquet('datasets/X_train.parquet')
        X_test = pd.read_parquet('datasets/X_test.parquet')
        # y_train = pd.read_parquet('datasets/y_train.parquet')
        # y_test = pd.read_parquet('datasets/y_test.parquet')
        COLS_TEXTO = ['artist_name', 'track_name', 'track_id', 'genre']
        cols_a_eliminar = [c for c in COLS_TEXTO if c in X_train.columns]

        X_train.drop(columns=cols_a_eliminar, inplace=True)
        X_test.drop(columns=cols_a_eliminar, inplace=True)
        clust_feats = ['instrumentalness', 'speechiness', 'danceability', 'valence', 'tempo']
        X_train= X_train[clust_feats]
        X_test= X_test[clust_feats]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logging.info("Datos de entrenamiento y prueba cargados y escalados correctamente")

        return X_train_scaled, X_test_scaled

    except Exception as e:
        logging.error(f"Error en la carga de datos de entrenamiento y prueba: {e}")


def entrenamiento(x_train):
    try:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(x_train)
        score = silhouette_score(x_train, kmeans.labels_)
        score_wcss = kmeans.inertia_
        logging.info(f"Modelo KMeans entrenado correctamente con silhouette score: {score}")
        logging.info(f"  - WCSS (Inercia): {score_wcss:.2f}")
        RUTA_MODELO = 'modelos/'
        os.makedirs(RUTA_MODELO, exist_ok=True)
        nombre_modelo = os.path.join(RUTA_MODELO, 'kmeans_spotify_model.pkl')
        joblib.dump(kmeans, nombre_modelo)
        logging.info(f"Modelo K-Means guardado en: {nombre_modelo}")


    except Exception as e:
        logging.error(f"Error en entrenamiento: {e}")

if __name__ == "__main__":
    iniciar_logger()
    x_train,x_test=Cargar_Datos_Ruta()
    entrenamiento(x_train)
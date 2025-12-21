# src/Entrenamiento.py
import os
import argparse
import logging
import pandas as pd
import joblib
import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def iniciar_logger(logs_dir: str):
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logs_dir, "Entrenamiento.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def cargar_datos(input_dir: str):
    X_train = pd.read_parquet(os.path.join(input_dir, "X_train.parquet"))
    X_test = pd.read_parquet(os.path.join(input_dir, "X_test.parquet"))

    cols_texto = ["artist_name", "track_name", "track_id", "genre"]
    X_train.drop(columns=[c for c in cols_texto if c in X_train], inplace=True)
    X_test.drop(columns=[c for c in cols_texto if c in X_test], inplace=True)

    feats = ["instrumentalness", "speechiness", "danceability", "valence", "tempo"]
    X_train = X_train[feats]
    X_test = X_test[feats]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, scaler

def entrenar(x_train, scaler, output_dir: str, k: int, random_state: int):
    os.makedirs(output_dir, exist_ok=True)

    inicio = time.time()

    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    model.fit(x_train)

    joblib.dump(model, os.path.join(output_dir, "kmeans_spotify_model.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    duracion = time.time() - inicio
    sil = silhouette_score(x_train, model.labels_) if k > 1 else 0

    logging.info(
        f"Modelo entrenado | k={k} | silhouette={sil:.4f} | {duracion:.2f}s"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--logs_dir", required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    iniciar_logger(args.logs_dir)

    x_train, scaler = cargar_datos(args.input_dir)
    entrenar(x_train, scaler, args.output_dir, args.k, args.random_state)

if __name__ == "__main__":
    main()

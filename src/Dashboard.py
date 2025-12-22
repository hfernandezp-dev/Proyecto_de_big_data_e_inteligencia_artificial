import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
from pathlib import Path

# Configuración visual
st.set_page_config(page_title="Spotify MLOps Dashboard", layout="wide")
st.title("Panel de Control: Spotify Clustering")

# Definición de rutas (ajustadas a tu estructura de carpetas)
BASE_DIR = Path(__file__).resolve().parent
RUTA_METRICAS = BASE_DIR / "modelos" / "metrics_history.json"
RUTA_DATASET = BASE_DIR / "datasets" / "canciones_clusterizadas.csv"

# --- VISTA 1: MÉTRICAS DE CALIDAD (Modelo) ---
st.header("Evolución de Calidad del Modelo")

if RUTA_METRICAS.exists():
    with open(RUTA_METRICAS, 'r') as f:
        data = json.load(f)
    df_metrics = pd.DataFrame(data)
    df_metrics['timestamp'] = pd.to_datetime(df_metrics['timestamp'])

    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Gráfico de Silhouette Score (Evolución temporal)
        fig_sil = px.line(df_metrics, x='timestamp', y='silhouette', 
                          title='Estabilidad del Silhouette Score',
                          markers=True, color_discrete_sequence=['#1DB954'])
        st.plotly_chart(fig_sil, use_container_width=True)
    
    with col2:
        # KPI Actual
        ultima_sil = df_metrics['silhouette'].iloc[-1]
        st.metric("Silhouette Actual", f"{ultima_sil:.4f}")
        # Alerta visual (Punto extra en rúbrica)
        if ultima_sil < 0.40:
            st.error("Calidad por debajo del umbral (0.40)")
        else:
            st.success("Modelo Saludable")
else:
    st.warning("No se encontró el historial de métricas. Ejecuta Entrenamiento.py primero.")

# --- VISTA 2: VISTA DE NEGOCIO (Catálogo) ---
st.markdown("---")
st.header("Distribución del Catálogo Musical")

if RUTA_DATASET.exists():
    df_cat = pd.read_csv(RUTA_DATASET)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Distribución de canciones por cluster
        fig_pie = px.pie(df_cat, names='cluster', title='Proporción de Canciones por Cluster',
                         color_discrete_sequence=px.colors.sequential.Greens_r)
        st.plotly_chart(fig_pie)
        
    with col_b:
        st.subheader("Muestra del Dataset Clusterizado")
        st.dataframe(df_cat[['artist_name', 'track_name', 'cluster']].head(10))
else:
    st.info("El dataset clusterizado aparecerá aquí después de ejecutar la API (main.py).")

# --- VISTA 3: SALUD DEL SERVICIO (Logs) ---
st.markdown("---")
st.header("Auditoría de Eventos (Logs)")
base_dir_api=Path(__file__).resolve().parent.parent
RUTA_LOGS = BASE_DIR / "FastAPIProject" / "logs" / "API.log"

if RUTA_LOGS.exists():
    with open(RUTA_LOGS, "r") as f:
        logs = f.readlines()
        # Mostrar los últimos 10 eventos
        st.text_area("Últimos eventos registrados:", "".join(logs[-10:]), height=200)
else:
    st.info("Aún no hay logs registrados por la API.")
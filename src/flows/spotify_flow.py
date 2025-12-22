from prefect import flow, task, get_run_logger
import subprocess
import sys
from pathlib import Path


# =========================
# Resolución robusta de rutas
# =========================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = None
for parent in CURRENT_FILE.parents:
    if (parent / "src" / "Pipeline.py").exists():
        PROJECT_ROOT = parent
        break

if PROJECT_ROOT is None:
    raise RuntimeError(" No se pudo localizar la raíz del proyecto")

SRC_DIR = PROJECT_ROOT / "src"

PIPELINE_SCRIPT = SRC_DIR / "Pipeline.py"
TRAINING_SCRIPT = SRC_DIR / "Entrenamiento.py"

DATASETS_DIR = SRC_DIR / "datasets"
MODELOS_DIR = SRC_DIR / "modelos"
LOGS_DIR = SRC_DIR / "logs"


# =========================
# Tasks
# =========================

@task(name="Ejecutar Pipeline de Datos", retries=2, retry_delay_seconds=10)
def run_pipeline():
    logger = get_run_logger()
    logger.info(f" Ejecutando Pipeline: {PIPELINE_SCRIPT}")

    cmd = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--input_csv", str(DATASETS_DIR / "spotify_data.csv"),
        "--output_dir", str(DATASETS_DIR),
        "--logs_dir", str(LOGS_DIR)
    ]

    result = subprocess.run(
        cmd,
        cwd=str(SRC_DIR),
        capture_output=True,
        text=True
    )

    logger.info(result.stdout)

    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(" Falló Pipeline.py")


@task(name="Ejecutar Entrenamiento", retries=1)
def run_training():
    logger = get_run_logger()
    logger.info(f" Ejecutando Entrenamiento: {TRAINING_SCRIPT}")

    cmd = [
        sys.executable,
        str(TRAINING_SCRIPT),
        "--input_dir", str(DATASETS_DIR),
        "--output_dir", str(MODELOS_DIR),
        "--logs_dir", str(LOGS_DIR),
        "--k", "4"
    ]

    result = subprocess.run(
        cmd,
        cwd=str(SRC_DIR),
        capture_output=True,
        text=True
    )

    logger.info(result.stdout)

    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(" Falló Entrenamiento.py")


# =========================
# Flow
# =========================

@flow(name="spotify_pipeline_flow")
def spotify_pipeline_flow():
    logger = get_run_logger()
    logger.info(" Iniciando pipeline Spotify con Prefect")

    run_pipeline()
    run_training()

    logger.info("✅ Pipeline Spotify finalizado correctamente")


if __name__ == "__main__":
    spotify_pipeline_flow()

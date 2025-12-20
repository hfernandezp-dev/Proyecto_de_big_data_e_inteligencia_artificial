from prefect import flow, task, get_run_logger
import subprocess
import sys
import os
from pathlib import Path


# Buscar la raíz real del proyecto (la que contiene src/)
CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = None
for parent in CURRENT_FILE.parents:
    if (parent / "src" / "Pipeline.py").exists():
        PROJECT_ROOT = parent
        break

if PROJECT_ROOT is None:
    raise RuntimeError("No se pudo localizar la raíz del proyecto")


PIPELINE_SCRIPT = PROJECT_ROOT / "src" / "Pipeline.py"
TRAINING_SCRIPT = PROJECT_ROOT / "src" / "Entrenamiento.py"


def run_pipeline_script():
    base_dir = Path(os.getenv("SPOTIFY_ARTIFACTS_DIR", "C:/prefect_artifacts/spotify"))

    cmd = [
        "python",
        "src/Pipeline.py",
        "--base-dir", str(base_dir),
        "--input-csv", str(base_dir / "raw/spotify_data.csv"),
        "--output-clean-csv", "datasets/spotify_data_clean.csv",
        "--sample-size", "2000",
        "--test-size", "0.2",
        "--random-state", "42",
    ]

    subprocess.run(cmd, check=True)


@task(name="Ejecutar Entrenamiento")
def run_training_script():
    logger = get_run_logger()
    logger.info(f"Ejecutando {TRAINING_SCRIPT}")

    result = subprocess.run(
        [sys.executable, str(TRAINING_SCRIPT)],
        cwd=str(TRAINING_SCRIPT.parent),  # 🔥 OBLIGATORIO
        capture_output=True,
        text=True
    )

    logger.info(result.stdout)

    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError("Falló Entrenamiento.py")


@flow(name="spotify_scripts_pipeline")
def spotify_pipeline_scripts():
    logger = get_run_logger()
    logger.info("🚀 Ejecutando pipeline basado en scripts")

    run_pipeline_script()
    run_training_script()

    logger.info("✅ Pipeline finalizado correctamente")


if __name__ == "__main__":
    spotify_pipeline_scripts()

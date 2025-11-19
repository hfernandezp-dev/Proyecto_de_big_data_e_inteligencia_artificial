import sys
from pathlib import Path

# Carpeta raíz del proyecto (donde están src/ y tests/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# La añadimos al PYTHONPATH
sys.path.insert(0, str(ROOT_DIR))
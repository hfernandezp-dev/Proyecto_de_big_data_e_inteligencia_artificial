# Dockerfile.api
FROM python:3.11-slim

WORKDIR /app

# Copiar requirements y luego instalar
COPY environment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto
COPY . .

# Establecer PYTHONPATH para imports
ENV PYTHONPATH=/app

# Exponer puerto de FastAPI
EXPOSE 8000

# Comando para ejecutar la API con hot-reload
CMD ["uvicorn", "FastAPIProject.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

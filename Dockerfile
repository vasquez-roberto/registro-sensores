FROM python:3.10-slim

# Directorio de trabajo en el contenedor
WORKDIR /app

# Copiar archivos de dependencias e instalarlas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY . .

# Comando por defecto al ejecutar el contenedor
CMD ["python", "script.py"]
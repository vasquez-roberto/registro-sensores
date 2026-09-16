import csv
import os
from pathlib import Path

import requests


def cargar_env(ruta_env):
    if not ruta_env.exists():
        return

    for linea in ruta_env.read_text(encoding="utf-8-sig").splitlines():
        linea = linea.strip()

        if not linea or linea.startswith("#") or "=" not in linea:
            continue

        clave, valor = linea.split("=", 1)
        os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))


cargar_env(Path(__file__).resolve().with_name(".env"))

API_KEY = os.getenv("API_KEY_PURPLEAIR")
OUTPUT_CSV = "sensores_detectados.csv"
SENSORES_BLOQUEADOS = {121825}

params = {
    "fields": "sensor_index,name,latitude,longitude,model,hardware,location_type",
    "nwlat": 25.78,
    "nwlng": -100.40,
    "selat": 25.60,
    "selng": -100.10,
}

headers = {"X-API-Key": API_KEY}
url = "https://api.purpleair.com/v1/sensors"


def obtener_sensores():
    if not API_KEY:
        print("Error: falta API_KEY_PURPLEAIR en el archivo .env.")
        return []

    response = requests.get(url, headers=headers, params=params, timeout=30)

    if response.status_code != 200:
        print(
            f"Error al consultar sensores: "
            f"{response.status_code} -> {response.text}"
        )
        return []

    resultado = response.json()
    campos = resultado["fields"]
    datos = resultado["data"]

    sensores = []

    for fila in datos:
        sensor = dict(zip(campos, fila))
        sensor_id = int(sensor["sensor_index"])

        if sensor_id in SENSORES_BLOQUEADOS:
            print(f"Sensor bloqueado omitido: {sensor_id}")
            continue

        sensores.append(sensor)

    return sensores


def guardar_csv(sensores):
    if not sensores:
        print("No se encontraron sensores.")
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=sensores[0].keys())
        writer.writeheader()
        writer.writerows(sensores)

    print(f"{len(sensores)} sensores guardados en {OUTPUT_CSV}")


if __name__ == "__main__":
    sensores = obtener_sensores()
    guardar_csv(sensores)
import csv
import requests

API_KEY = "C7B070E7-AAFE-11F1-9E30-4201AC1DC129"
OUTPUT_CSV = "sensores_detectados.csv"

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
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print(
            f"Error al consultar sensores: {response.status_code} -> {response.text}"
        )
        return []

    resultado = response.json()
    campos = resultado["fields"]
    datos = resultado["data"]

    sensores = []
    for fila in datos:
        sensor = dict(zip(campos, fila))
        sensores.append(sensor)

    return sensores


def guardar_csv(sensores):
    if not sensores:
        print("No se encontraron sensores.")
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sensores[0].keys())
        writer.writeheader()
        writer.writerows(sensores)

    print(f"{len(sensores)} sensores guardados en {OUTPUT_CSV}")


if __name__ == "__main__":
    sensores = obtener_sensores()
    guardar_csv(sensores)
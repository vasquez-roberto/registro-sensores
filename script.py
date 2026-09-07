import csv
import json
import os
import requests

API_KEY = os.getenv(
    "API_KEY_PURPLEAIR", "C7B070E7-AAFE-11F1-9E30-4201AC1DC129"
)
INPUT_CSV = "sensores_detectados.csv"
OUTPUT_GEOJSON = "sensores.geojson"


def cargar_indices_sensores():
    if not os.path.exists(INPUT_CSV):
        print(
            f"Error: {INPUT_CSV} no existe. Ejecuta sensor_location.py primero."
        )
        return []

    indices = []
    with open(INPUT_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            indices.append(row["sensor_index"])
    return indices


def obtener_lecturas_sensores(indices):
    if not indices:
        return []

    url = "https://api.purpleair.com/v1/sensors"
    # Campos estándar de medición de calidad de aire en PurpleAir
    params = {
        "show_only": ",".join(indices),
        "fields": "sensor_index,name,latitude,longitude,pm2.5,pm10.0,humidity,temperature",
    }
    headers = {"X-API-Key": API_KEY}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print(f"Error al leer datos: {response.status_code} -> {response.text}")
        return []

    data = response.json()
    campos = data["fields"]
    registros = data["data"]

    features = []
    for reg in registros:
        d = dict(zip(campos, reg))
        lat = d.get("latitude")
        lon = d.get("longitude")

        if lat is not None and lon is not None:
            feature = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "sensor_index": d.get("sensor_index"),
                    "name": d.get("name"),
                    "pm2_5": d.get("pm2.5"),
                    "pm10_0": d.get("pm10.0"),
                    "temperatura": d.get("temperature"),
                    "humedad": d.get("humidity"),
                },
            }
            features.append(feature)

    return features


def guardar_geojson(features):
    geojson = {"type": "FeatureCollection", "features": features}

    with open(OUTPUT_GEOJSON, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=4, ensure_ascii=False)

    print(
        f"GeoJSON generado exitosamente en {OUTPUT_GEOJSON} con {len(features)} elementos."
    )


if __name__ == "__main__":
    indices = cargar_indices_sensores()
    features = obtener_lecturas_sensores(indices)
    guardar_geojson(features)
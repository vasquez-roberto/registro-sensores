from datetime import datetime, timezone
import json
import os
from pathlib import Path

import geojson
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from scipy.spatial import Delaunay
from shapely.geometry import Point, mapping


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
CSV_FILE = "sensores_detectados.csv"
SALIDA_GEOJSON_SENSORES = "sensores.geojson"
SALIDA_GEOJSON_COLONIAS_PM25 = "AQ_PM25.geojson"
SALIDA_GEOJSON_COLONIAS_PM10 = "AQ_PM10.geojson"
ARCHIVO_SHP_COLONIAS = "shp/2025_1_19_A.shp"
CAMPOS = "pm1.0,pm2.5"

# Límites de cribado basados en el extremo superior de las tablas AQI de EPA.
# No son límites físicos: un evento extraordinario debe revisarse manualmente.
MIN_PM = 0.0
MAX_PM25 = 500.4
MAX_PM10 = 604.0


def leer_csv(ruta):
    df = pd.read_csv(ruta)
    df = df.dropna(subset=["latitude", "longitude", "sensor_index"])
    df["sensor_index"] = df["sensor_index"].astype(int)
    return df


def consultar_sensor(sensor_index):
    if not API_KEY:
        raise RuntimeError("Falta API_KEY_PURPLEAIR en .env")

    url = f"https://api.purpleair.com/v1/sensors/{sensor_index}?fields={CAMPOS}"
    try:
        respuesta = requests.get(url, headers={"X-API-Key": API_KEY}, timeout=15)
        respuesta.raise_for_status()
        sensor = respuesta.json().get("sensor", {})
        return sensor.get("pm1.0"), sensor.get("pm2.5")
    except requests.RequestException as error:
        print(f"No se pudo consultar {sensor_index}: {error}")
        return None, None


def clasificar_calidad_aire_pm25(valor):
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "Sin datos"
    if valor <= 15:
        return "Bueno"
    if valor <= 33:
        return "Aceptable"
    if valor <= 79:
        return "Mala"
    if valor <= 130:
        return "Muy alta"
    return "Extremadamente mala"


def clasificar_calidad_aire_pm10(valor):
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "Sin datos"
    if valor <= 45:
        return "Bueno"
    if valor <= 60:
        return "Aceptable"
    if valor <= 132:
        return "Mala"
    if valor <= 213:
        return "Muy alta"
    return "Extremadamente mala"


def lecturas_validas(sensor_id, pm10, pm25):
    """Descarta faltantes, valores no numéricos y valores fuera del cribado."""
    try:
        pm10, pm25 = float(pm10), float(pm25)
    except (TypeError, ValueError):
        print(f"Sensor {sensor_id} descartado: lectura no numérica.")
        return None

    if not np.isfinite(pm10) or not np.isfinite(pm25):
        print(f"Sensor {sensor_id} descartado: lectura no finita.")
        return None

    if not (MIN_PM <= pm10 <= MAX_PM10 and MIN_PM <= pm25 <= MAX_PM25):
        print(
            f"Sensor {sensor_id} descartado por valor fuera de rango: "
            f"PM10={pm10}, PM2.5={pm25}"
        )
        return None

    return pm10, pm25


def crear_geojson(df, timestamp):
    features, puntos, valores_pm25, valores_pm10, historico = [], [], [], [], []

    for _, fila in df.iterrows():
        sensor_id = int(fila["sensor_index"])
        pm10, pm25 = consultar_sensor(sensor_id)
        if pm10 is None or pm25 is None:
            continue

        valores = lecturas_validas(sensor_id, pm10, pm25)
        if valores is None:
            continue
        pm10, pm25 = valores

        coordenadas = [float(fila["longitude"]), float(fila["latitude"])]
        nombre = fila.get("name", "")
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": coordenadas},
            "properties": {
                "sensor_index": sensor_id,
                "name": nombre,
                "pm1_0": pm10,
                "pm2_5": pm25,
                "AQ PM 2.5": clasificar_calidad_aire_pm25(pm25),
                "AQ PM 10": clasificar_calidad_aire_pm10(pm10),
                "timestamp": timestamp,
            },
        })
        puntos.append(coordenadas)
        valores_pm25.append(pm25)
        valores_pm10.append(pm10)
        historico.append({
            "sensor_index": sensor_id, "name": nombre, "timestamp": timestamp,
            "pm1_0": pm10, "pm2_5": pm25,
        })

    with open(SALIDA_GEOJSON_SENSORES, "w", encoding="utf-8") as archivo:
        json.dump({"type": "FeatureCollection", "features": features}, archivo,
                  ensure_ascii=False, indent=2)
    print(f"GeoJSON de sensores generado ({len(features)} entidades): {SALIDA_GEOJSON_SENSORES}")

    nuevo = pd.DataFrame(historico)
    if os.path.exists("historico.csv"):
        try:
            nuevo = pd.concat([pd.read_csv("historico.csv"), nuevo], ignore_index=True)
        except Exception:
            pass
    nuevo.to_csv("historico.csv", index=False, encoding="utf-8")

    return np.array(puntos), np.array(valores_pm25), np.array(valores_pm10)


def cargar_datos_colonias_shp(archivo_shp):
    """Lee el CRS del .prj MEXICO_ITRF_2008_LCC y lo convierte a WGS84."""
    gdf = gpd.read_file(archivo_shp)
    if gdf.crs is None:
        raise ValueError("El SHP no tiene CRS definido en su archivo .prj.")

    gdf = gdf.to_crs("EPSG:4326")
    nombre_columna = next(col for col in gdf.columns if col != "geometry")

    return [
        {"nombre": str(fila[nombre_columna]), "geometry": fila.geometry}
        for _, fila in gdf.iterrows()
        if fila.geometry is not None and not fila.geometry.is_empty
    ]


def interpolar_lineal(punto, triangulo_indices, puntos, valores):
    v0, v1, v2 = puntos[triangulo_indices]
    z0, z1, z2 = valores[triangulo_indices]
    try:
        pesos = np.linalg.solve(
            np.array([v1 - v0, v2 - v0]).T,
            punto - v0,
        )
        return (1 - pesos[0] - pesos[1]) * z0 + pesos[0] * z1 + pesos[1] * z2
    except np.linalg.LinAlgError:
        return None


def generar_geojson_colonias(nombre_archivo, colonias_data, puntos_data,
                              valores_puntos, contaminante, timestamp):
    try:
        triangulacion = Delaunay(puntos_data)
    except Exception as error:
        print(f"Error Delaunay: {error}")
        triangulacion = None

    features = []
    for colonia in colonias_data:
        geom = colonia["geometry"]
        valores_en_colonia = [
            valores_puntos[i] for i, (lon, lat) in enumerate(puntos_data)
            if geom.contains(Point(lon, lat))
        ]
        if valores_en_colonia:
            valor = float(np.mean(valores_en_colonia))
        elif triangulacion is None:
            valor = None
        else:
            centroide = geom.centroid
            punto = np.array([centroide.x, centroide.y])
            indice = triangulacion.find_simplex(punto)
            valor = (interpolar_lineal(punto, triangulacion.simplices[indice],
                                       puntos_data, valores_puntos)
                     if indice != -1 else None)

        valor = round(float(valor), 2) if valor is not None and np.isfinite(valor) else None
        features.append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {
                "nombre": colonia["nombre"],
                "valor_interpolado": valor,
                "AQ": (clasificar_calidad_aire_pm25(valor) if contaminante == "pm2_5"
                       else clasificar_calidad_aire_pm10(valor)),
                "timestamp": timestamp,
            },
        })

    with open(nombre_archivo, "w", encoding="utf-8") as archivo:
        json.dump({"type": "FeatureCollection", "features": features}, archivo,
                  ensure_ascii=False, indent=2, allow_nan=False)
    print(f"GeoJSON reproyectado generado ({len(features)} entidades): {nombre_archivo}")


if __name__ == "__main__":
    ejecucion = datetime.now(timezone.utc).isoformat()
    sensores = leer_csv(CSV_FILE)
    puntos, pm25, pm10 = crear_geojson(sensores, ejecucion)

    if len(puntos) >= 3 and os.path.exists(ARCHIVO_SHP_COLONIAS):
        colonias = cargar_datos_colonias_shp(ARCHIVO_SHP_COLONIAS)
        generar_geojson_colonias(SALIDA_GEOJSON_COLONIAS_PM25, colonias, puntos,
                                 pm25, "pm2_5", ejecucion)
        generar_geojson_colonias(SALIDA_GEOJSON_COLONIAS_PM10, colonias, puntos,
                                 pm10, "pm10", ejecucion)
    else:
        print("No se generaron capas de colonias: faltan sensores o el SHP.")

    print("Proceso completado.")

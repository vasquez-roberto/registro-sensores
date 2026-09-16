from datetime import datetime, timezone
import json
import os
from pathlib import Path

import geojson
import numpy as np
import pandas as pd
import requests
from scipy.spatial import Delaunay
import shapefile
from shapely.geometry import Point, shape


def cargar_env(ruta_env):
    if not ruta_env.exists():
        return

    for linea in ruta_env.read_text(encoding="utf-8-sig").splitlines():
        linea = linea.strip()

        if not linea or linea.startswith("#") or "=" not in linea:
            continue

        clave, valor = linea.split("=", 1)
        os.environ.setdefault(
            clave.strip(),
            valor.strip().strip('"').strip("'"),
        )


cargar_env(Path(__file__).resolve().with_name(".env"))

print("INICIANDO SCRIPT...")

API_KEY = os.getenv("API_KEY_PURPLEAIR")

if not API_KEY:
    raise RuntimeError(
        "No se encontró API_KEY_PURPLEAIR en el archivo .env."
    )

CSV_FILE = "sensores_detectados.csv"
SALIDA_GEOJSON_SENSORES = "sensores.geojson"
SALIDA_GEOJSON_COLONIAS_PM25 = "AQ_PM25.geojson"
SALIDA_GEOJSON_COLONIAS_PM10 = "AQ_PM10.geojson"
ARCHIVO_SHP_COLONIAS = "shp/2025_1_19_A.shp"
CAMPOS = "pm1.0,pm2.5"

SENSORES_BLOQUEADOS = {121825}


def leer_csv(ruta):
    df = pd.read_csv(ruta)
    df = df.dropna(subset=["latitude", "longitude", "sensor_index"])
    df["sensor_index"] = df["sensor_index"].astype(int)

    df = df[~df["sensor_index"].isin(SENSORES_BLOQUEADOS)]

    return df


def consultar_sensor(sensor_index):
    if int(sensor_index) in SENSORES_BLOQUEADOS:
        return None, None

    url = f"https://api.purpleair.com/v1/sensors/{sensor_index}?fields={CAMPOS}"
    headers = {"X-API-Key": API_KEY} if API_KEY else {}

    try:
        response = requests.get(url, headers=headers, timeout=10)
    except Exception:
        return None, None

    if response.status_code == 200:
        data = response.json().get("sensor", {})
        return data.get("pm1.0"), data.get("pm2.5")

    return None, None


def clasificar_calidad_aire_pm25(pm25):
    if pm25 is None or (isinstance(pm25, float) and np.isnan(pm25)):
        return "Sin datos"

    try:
        pm25 = float(pm25)
    except Exception:
        return "Sin datos"

    if pm25 <= 15:
        return "Bueno"
    elif pm25 <= 33:
        return "Aceptable"
    elif pm25 <= 79:
        return "Mala"
    elif pm25 <= 130:
        return "Muy alta"

    return "Extremadamente mala"


def clasificar_calidad_aire_pm10(pm10):
    if pm10 is None or (isinstance(pm10, float) and np.isnan(pm10)):
        return "Sin datos"

    try:
        pm10 = float(pm10)
    except Exception:
        return "Sin datos"

    if pm10 <= 45:
        return "Bueno"
    elif pm10 <= 60:
        return "Aceptable"
    elif pm10 <= 132:
        return "Mala"
    elif pm10 <= 213:
        return "Muy alta"

    return "Extremadamente mala"


def crear_geojson(df, timestamp):
    features, puntos, valores_pm25, valores_pm10 = [], [], [], []
    datos_historicos = []

    for _, fila in df.iterrows():
        sensor_id = int(fila["sensor_index"])

        if sensor_id in SENSORES_BLOQUEADOS:
            continue

        print(f"Consultando sensor {sensor_id}...")
        pm10, pm25 = consultar_sensor(sensor_id)

        if pm10 is not None and pm25 is not None:
            props = {
                "sensor_index": sensor_id,
                "name": fila.get("name", ""),
                "pm1_0": pm10,
                "pm2_5": pm25,
                "AQ PM 2.5": clasificar_calidad_aire_pm25(pm25),
                "AQ PM 10": clasificar_calidad_aire_pm10(pm10),
                "timestamp": timestamp,
            }

            coords = (float(fila["longitude"]), float(fila["latitude"]))

            features.append(
                geojson.Feature(
                    geometry=geojson.Point(coords),
                    properties=props,
                )
            )

            puntos.append(coords)
            valores_pm25.append(float(pm25))
            valores_pm10.append(float(pm10))

            datos_historicos.append(
                {
                    "sensor_index": sensor_id,
                    "name": fila.get("name", ""),
                    "timestamp": timestamp,
                    "pm1_0": pm10,
                    "pm2_5": pm25,
                }
            )

    with open(SALIDA_GEOJSON_SENSORES, "w", encoding="utf-8") as archivo:
        geojson.dump(geojson.FeatureCollection(features), archivo, indent=2)

    print(f"GeoJSON de sensores generado: {SALIDA_GEOJSON_SENSORES}")

    df_nuevo = pd.DataFrame(datos_historicos)

    if os.path.exists("historico.csv"):
        try:
            df_existente = pd.read_csv("historico.csv")
            df_total = pd.concat([df_existente, df_nuevo], ignore_index=True)
        except Exception:
            df_total = df_nuevo
    else:
        df_total = df_nuevo

    if "sensor_index" in df_total.columns:
        df_total = df_total[
            ~df_total["sensor_index"].isin(SENSORES_BLOQUEADOS)
        ]

    df_total.to_csv("historico.csv", index=False, encoding="utf-8")

    return (
        np.array(puntos),
        np.array(valores_pm25),
        np.array(valores_pm10),
    )


def cargar_datos_colonias_shp(archivo_shp):
    sf = shapefile.Reader(archivo_shp, encoding="utf-8")
    colonias = []

    for shape_record in sf.iterShapeRecords():
        geometry = shape(shape_record.shape.__geo_interface__)
        nombre_colonia = shape_record.record[0]

        colonias.append(
            {
                "nombre": nombre_colonia,
                "geometry": geometry,
            }
        )

    return colonias


def interpolar_lineal(punto, triangulo_indices, puntos, valores):
    v0, v1, v2 = puntos[triangulo_indices]
    z0, z1, z2 = valores[triangulo_indices]

    delta1 = v1 - v0
    delta2 = v2 - v0
    delta_p = punto - v0

    try:
        A = np.array([delta1, delta2]).T
        w = np.linalg.solve(A, delta_p)

        return (
            (1 - w[0] - w[1]) * z0
            + w[0] * z1
            + w[1] * z2
        )
    except np.linalg.LinAlgError:
        return None


def generar_geojson_colonias(
    nombre_archivo,
    colonias_data,
    puntos_data,
    valores_puntos,
    contaminante,
    timestamp,
):
    try:
        tri = Delaunay(puntos_data)
    except Exception as error:
        print(f"Error Delaunay: {error}")
        tri = None

    for colonia in colonias_data:
        geom = colonia["geometry"]

        valores_en_colonia = [
            valores_puntos[i]
            for i, (lon, lat) in enumerate(puntos_data)
            if geom.contains(Point(lon, lat))
        ]

        if valores_en_colonia:
            colonia["valor_interpolado"] = float(np.mean(valores_en_colonia))
        else:
            if tri is None:
                colonia["valor_interpolado"] = np.nan
                continue

            centroide = geom.centroid
            p_cent = np.array([centroide.x, centroide.y])
            idx = tri.find_simplex(p_cent)

            colonia["valor_interpolado"] = (
                interpolar_lineal(
                    p_cent,
                    tri.simplices[idx],
                    puntos_data,
                    valores_puntos,
                )
                if idx != -1
                else np.nan
            )

    geo_json_data = {
        "type": "FeatureCollection",
        "metadata": {"ultima_ejecucion_utc": timestamp},
        "features": [],
    }

    for colonia in colonias_data:
        geom = colonia["geometry"]

        if not geom.is_valid or geom.is_empty:
            continue

        try:
            if geom.geom_type == "Polygon":
                geometry = {
                    "type": "Polygon",
                    "coordinates": [list(geom.exterior.coords)],
                }

            elif geom.geom_type == "MultiPolygon":
                geometry = {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [list(poligono.exterior.coords)]
                        for poligono in geom.geoms
                    ],
                }

            else:
                continue

        except Exception:
            continue

        valor = colonia.get("valor_interpolado")

        valor_exportado = (
            round(float(valor), 2)
            if valor is not None and not np.isnan(valor)
            else None
        )

        geo_json_data["features"].append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "nombre": colonia["nombre"],
                    "valor_interpolado": valor_exportado,
                    "AQ": (
                        clasificar_calidad_aire_pm25(valor_exportado)
                        if contaminante == "pm2_5"
                        else clasificar_calidad_aire_pm10(valor_exportado)
                    ),
                },
            }
        )

    with open(nombre_archivo, "w", encoding="utf-8") as archivo:
        json.dump(geo_json_data, archivo, ensure_ascii=False, indent=2)

    print(f"GeoJSON generado: {nombre_archivo}")


if __name__ == "__main__":
    timestamp_ejecucion = datetime.now(timezone.utc).isoformat()

    df_sensores = leer_csv(CSV_FILE)

    puntos_data, valores_pm25, valores_pm10 = crear_geojson(
        df_sensores,
        timestamp_ejecucion,
    )

    if puntos_data.size > 0 and os.path.exists(ARCHIVO_SHP_COLONIAS):
        colonias_pm25 = cargar_datos_colonias_shp(ARCHIVO_SHP_COLONIAS)
        colonias_pm10 = cargar_datos_colonias_shp(ARCHIVO_SHP_COLONIAS)

        generar_geojson_colonias(
            SALIDA_GEOJSON_COLONIAS_PM25,
            colonias_pm25,
            puntos_data,
            valores_pm25,
            "pm2_5",
            timestamp_ejecucion,
        )

        generar_geojson_colonias(
            SALIDA_GEOJSON_COLONIAS_PM10,
            colonias_pm10,
            puntos_data,
            valores_pm10,
            "pm10",
            timestamp_ejecucion,
        )

    try:
        import geopandas as gpd

        if os.path.exists("historico.csv") and os.path.exists(CSV_FILE):
            df_hist = pd.read_csv("historico.csv")
            df_det = pd.read_csv(CSV_FILE)

            if "sensor_index" in df_hist.columns:
                df_hist = df_hist[
                    ~df_hist["sensor_index"].isin(SENSORES_BLOQUEADOS)
                ]

            if "sensor_index" in df_det.columns:
                df_det = df_det[
                    ~df_det["sensor_index"].isin(SENSORES_BLOQUEADOS)
                ]

            if "latitude" in df_det.columns and "longitude" in df_det.columns:
                df_comb = df_hist.merge(
                    df_det[["name", "latitude", "longitude"]],
                    on="name",
                    how="left",
                ).dropna(subset=["latitude", "longitude"])

                gdf = gpd.GeoDataFrame(
                    df_comb,
                    geometry=gpd.points_from_xy(
                        df_comb.longitude,
                        df_comb.latitude,
                    ),
                    crs="EPSG:4326",
                )

                gdf.to_file(
                    "historico_combinado.geojson",
                    driver="GeoJSON",
                )

    except Exception as error:
        print(f"Aviso geopandas: {error}")

    print("Proceso completado exitosamente.")
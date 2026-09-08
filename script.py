from datetime import datetime, timezone
import json
import os
import geojson
import numpy as np
import pandas as pd
import requests
from scipy.spatial import Delaunay
import shapefile
from shapely.geometry import Point, shape

API_KEY = os.getenv(
    "API_KEY_PURPLEAIR", "C7B070E7-AAFE-11F1-9E30-4201AC1DC129"
)
CSV_FILE = "sensores_detectados.csv"
SALIDA_GEOJSON_SENSORES = "sensores.geojson"
SALIDA_GEOJSON_COLONIAS_PM25 = "AQ_PM25.geojson"
SALIDA_GEOJSON_COLONIAS_PM10 = "AQ_PM10.geojson"
ARCHIVO_SHP_COLONIAS = "shp/2023_1_19_A.shp"
CAMPOS = "pm1.0,pm2.5"


def leer_csv(ruta):
    df = pd.read_csv(ruta)
    df = df.dropna(subset=["latitude", "longitude", "sensor_index"])
    df["sensor_index"] = df["sensor_index"].astype(int)
    return df


def consultar_sensor(sensor_index):
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
    else:
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
    else:
        return "Extremadamente mala"


def crear_geojson(df, timestamp):
    features, puntos, valores_pm25, valores_pm10 = [], [], [], []
    datos_historicos = []

    for _, fila in df.iterrows():
        pm10, pm25 = consultar_sensor(fila["sensor_index"])
        if pm10 is not None and pm25 is not None:
            props = {
                "sensor_index": int(fila["sensor_index"]),
                "name": fila.get("name", ""),
                "pm1_0": pm10,
                "pm2_5": pm25,
                "AQ PM 2.5": clasificar_calidad_aire_pm25(pm25),
                "AQ PM 10": clasificar_calidad_aire_pm10(pm10),
                "timestamp": timestamp,
            }
            coords = (float(fila["longitude"]), float(fila["latitude"]))
            features.append(
                geojson.Feature(geometry=geojson.Point(coords), properties=props)
            )
            puntos.append(coords)
            valores_pm25.append(float(pm25))
            valores_pm10.append(float(pm10))

            datos_historicos.append({
                "sensor_index": int(fila["sensor_index"]),
                "name": fila.get("name", ""),
                "timestamp": timestamp,
                "pm1_0": pm10,
                "pm2_5": pm25,
            })

    with open(SALIDA_GEOJSON_SENSORES, "w", encoding="utf-8") as f:
        geojson.dump(geojson.FeatureCollection(features), f, indent=2)
    print(f"GeoJSON de sensores generado: {SALIDA_GEOJSON_SENSORES}")

    df_nuevo = pd.DataFrame(datos_historicos)
    if os.path.exists("historico.csv"):
        try:
            df_ex = pd.read_csv("historico.csv")
            df_tot = pd.concat([df_ex, df_nuevo], ignore_index=True)
        except Exception:
            df_tot = df_nuevo
    else:
        df_tot = df_nuevo
    df_tot.to_csv("historico.csv", index=False, encoding="utf-8")

    return np.array(puntos), np.array(valores_pm25), np.array(valores_pm10)


def cargar_datos_colonias_shp(archivo_shp):
    sf = shapefile.Reader(archivo_shp, encoding="utf-8")
    colonias = []
    for shape_record in sf.iterShapeRecords():
        geometry = shape(shape_record.shape.__geo_interface__)
        nombre_colonia = shape_record.record[0]
        colonias.append({"nombre": nombre_colonia, "geometry": geometry})
    return colonias


def interpolar_lineal(punto, triangulo_indices, puntos, valores):
    v0, v1, v2 = puntos[triangulo_indices]
    z0, z1, z2 = valores[triangulo_indices]
    delta1, delta2, delta_p = v1 - v0, v2 - v0, punto - v0
    try:
        A = np.array([delta1, delta2]).T
        w = np.linalg.solve(A, delta_p)
        return (1 - w[0] - w[1]) * z0 + w[0] * z1 + w[1] * z2
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
    except Exception as e:
        print(f"Error Delaunay: {e}")
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
                interpolar_lineal(p_cent, tri.simplices[idx], puntos_data, valores_puntos)
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
                    "coordinates": [[list(p.exterior.coords)] for p in geom.geoms],
                }
            else:
                continue
        except Exception:
            continue

        val = colonia.get("valor_interpolado")
        val_exp = (
            round(float(val), 2)
            if val is not None and not np.isnan(val)
            else None
        )

        geo_json_data["features"].append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "nombre": colonia["nombre"],
                "valor_interpolado": val_exp,
                "AQ": (
                    clasificar_calidad_aire_pm25(val_exp)
                    if contaminante == "pm2_5"
                    else clasificar_calidad_aire_pm10(val_exp)
                ),
            },
        })

    with open(nombre_archivo, "w", encoding="utf-8") as f:
        json.dump(geo_json_data, f, ensure_ascii=False, indent=2)
    print(f"GeoJSON generado: {nombre_archivo}")


if __name__ == "__main__":
    timestamp_ejecucion = datetime.now(timezone.utc).isoformat()
    df_sensores = leer_csv(CSV_FILE)
    puntos_data, valores_pm25, valores_pm10 = crear_geojson(
        df_sensores, timestamp_ejecucion
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
            df_hist, df_det = pd.read_csv("historico.csv"), pd.read_csv(CSV_FILE)
            if "latitude" in df_det.columns and "longitude" in df_det.columns:
                df_comb = df_hist.merge(
                    df_det[["name", "latitude", "longitude"]],
                    on="name",
                    how="left",
                ).dropna(subset=["latitude", "longitude"])
                gdf = gpd.GeoDataFrame(
                    df_comb,
                    geometry=gpd.points_from_xy(
                        df_comb.longitude, df_comb.latitude
                    ),
                    crs="EPSG:4326",
                )
                gdf.to_file("historico_combinado.geojson", driver="GeoJSON")
    except Exception as e:
        print(f"Aviso geopandas: {e}")

    print("Proceso completado exitosamente.")

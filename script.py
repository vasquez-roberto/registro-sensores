"""Genera GeoJSON de sensores y colonias interpoladas para ArcGIS.

El sensor 121825 se excluye antes de consultar la API, de la interpolacion
y de los archivos historicos.  Las colonias se recortan al casco convexo de
los sensores validos: asi los GeoJSON no incluyen zonas sin interpolacion.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import shapefile
from scipy.spatial import Delaunay
from shapely.geometry import MultiPoint, Point, mapping, shape
from shapely.ops import unary_union


def cargar_env(ruta_env):
    """Carga pares CLAVE=VALOR de un .env sin dependencias adicionales."""
    if not ruta_env.exists():
        return
    for linea in ruta_env.read_text(encoding="utf-8-sig").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, valor = linea.split("=", 1)
        clave, valor = clave.strip(), valor.strip().strip('"').strip("'")
        # Una variable definida en PowerShell tiene prioridad sobre el archivo .env.
        os.environ.setdefault(clave, valor)


# Lee el .env ubicado en la misma carpeta que este script.
cargar_env(Path(__file__).resolve().with_name(".env"))
API_KEY = os.getenv("API_KEY_PURPLEAIR")
CSV_FILE = "sensores_detectados.csv"
SALIDA_GEOJSON_SENSORES = "sensores.geojson"
SALIDA_GEOJSON_COLONIAS_PM25 = "AQ_PM25.geojson"
SALIDA_GEOJSON_COLONIAS_PM10 = "AQ_PM10.geojson"
ARCHIVO_SHP_COLONIAS = "shp/2025_1_19_A.shp"
CAMPOS = "pm10.0,pm2.5"

# Agregue otros IDs si en el futuro se deben excluir del analisis.
SENSORES_BLOQUEADOS = {121825}


def leer_csv(ruta):
    df = pd.read_csv(ruta)
    columnas = {"latitude", "longitude", "sensor_index"}
    faltantes = columnas.difference(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en {ruta}: {sorted(faltantes)}")

    df = df.dropna(subset=["latitude", "longitude", "sensor_index"]).copy()
    df["sensor_index"] = pd.to_numeric(df["sensor_index"], errors="coerce")
    df = df.dropna(subset=["sensor_index"])
    df["sensor_index"] = df["sensor_index"].astype(int)
    df = df[~df["sensor_index"].isin(SENSORES_BLOQUEADOS)].copy()
    return df


def consultar_sensor(sensor_index):
    if sensor_index in SENSORES_BLOQUEADOS:
        return None, None
    if not API_KEY:
        raise RuntimeError("Defina la variable de entorno API_KEY_PURPLEAIR.")

    url = f"https://api.purpleair.com/v1/sensors/{sensor_index}?fields={CAMPOS}"
    try:
        response = requests.get(url, headers={"X-API-Key": API_KEY}, timeout=15)
        response.raise_for_status()
        data = response.json().get("sensor", {})
        return data.get("pm10.0"), data.get("pm2.5")
    except (requests.RequestException, ValueError) as error:
        print(f"No se pudo consultar el sensor {sensor_index}: {error}")
        return None, None


def valor_numerico(valor):
    try:
        valor = float(valor)
    except (TypeError, ValueError):
        return None
    return valor if np.isfinite(valor) else None


def clasificar_calidad_aire_pm25(pm25):
    pm25 = valor_numerico(pm25)
    if pm25 is None:
        return "Sin datos"
    if pm25 <= 15:
        return "Bueno"
    if pm25 <= 33:
        return "Aceptable"
    if pm25 <= 79:
        return "Mala"
    if pm25 <= 130:
        return "Muy alta"
    return "Extremadamente mala"


def clasificar_calidad_aire_pm10(pm10):
    pm10 = valor_numerico(pm10)
    if pm10 is None:
        return "Sin datos"
    if pm10 <= 45:
        return "Bueno"
    if pm10 <= 60:
        return "Aceptable"
    if pm10 <= 132:
        return "Mala"
    if pm10 <= 213:
        return "Muy alta"
    return "Extremadamente mala"


def crear_geojson(df, timestamp):
    features, puntos, valores_pm25, valores_pm10, datos_historicos = [], [], [], [], []

    for _, fila in df.iterrows():
        sensor_idx = int(fila["sensor_index"])
        if sensor_idx in SENSORES_BLOQUEADOS:
            continue

        pm10, pm25 = consultar_sensor(sensor_idx)
        pm10, pm25 = valor_numerico(pm10), valor_numerico(pm25)
        if pm10 is None or pm25 is None:
            continue

        coords = [float(fila["longitude"]), float(fila["latitude"])]
        nombre = fila.get("name", "")
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": coords},
            "properties": {
                "sensor_index": sensor_idx,
                "name": nombre,
                "pm10": pm10,
                "pm2_5": pm25,
                "AQ_PM25": clasificar_calidad_aire_pm25(pm25),
                "AQ_PM10": clasificar_calidad_aire_pm10(pm10),
                "timestamp_utc": timestamp,
            },
        })
        puntos.append(coords)
        valores_pm25.append(pm25)
        valores_pm10.append(pm10)
        datos_historicos.append({
            "sensor_index": sensor_idx, "name": nombre, "timestamp": timestamp,
            "pm10": pm10, "pm2_5": pm25,
        })

    with open(SALIDA_GEOJSON_SENSORES, "w", encoding="utf-8") as archivo:
        json.dump({"type": "FeatureCollection", "features": features}, archivo,
                  ensure_ascii=False, indent=2, allow_nan=False)
    print(f"GeoJSON de sensores generado: {SALIDA_GEOJSON_SENSORES}")

    df_nuevo = pd.DataFrame(datos_historicos)
    if os.path.exists("historico.csv"):
        try:
            df_nuevo = pd.concat([pd.read_csv("historico.csv"), df_nuevo], ignore_index=True)
        except (OSError, pd.errors.ParserError) as error:
            print(f"No se pudo leer el historico anterior: {error}")
    if not df_nuevo.empty and "sensor_index" in df_nuevo:
        df_nuevo = df_nuevo[~df_nuevo["sensor_index"].isin(SENSORES_BLOQUEADOS)]
    df_nuevo.to_csv("historico.csv", index=False, encoding="utf-8")

    return np.asarray(puntos, dtype=float), np.asarray(valores_pm25, dtype=float), np.asarray(valores_pm10, dtype=float)


def cargar_datos_colonias_shp(archivo_shp):
    sf = shapefile.Reader(archivo_shp, encoding="utf-8")
    colonias = []
    for registro in sf.iterShapeRecords():
        geometria = shape(registro.shape.__geo_interface__)
        if not geometria.is_valid:
            geometria = geometria.buffer(0)
        if not geometria.is_empty:
            colonias.append({"nombre": str(registro.record[0]), "geometry": geometria})
    return colonias


def interpolar_lineal(punto, triangulo_indices, puntos, valores):
    vertices = puntos[triangulo_indices]
    matriz = np.column_stack((vertices[1] - vertices[0], vertices[2] - vertices[0]))
    try:
        pesos = np.linalg.solve(matriz, punto - vertices[0])
    except np.linalg.LinAlgError:
        return None
    return float((1 - pesos[0] - pesos[1]) * valores[triangulo_indices[0]] +
                 pesos[0] * valores[triangulo_indices[1]] +
                 pesos[1] * valores[triangulo_indices[2]])


def solo_poligonos(geometria):
    if geometria.geom_type in {"Polygon", "MultiPolygon"}:
        return geometria
    poligonos = [g for g in getattr(geometria, "geoms", []) if g.geom_type in {"Polygon", "MultiPolygon"}]
    return unary_union(poligonos) if poligonos else None


def generar_geojson_colonias(nombre_archivo, colonias_data, puntos_data,
                              valores_puntos, contaminante, timestamp):
    if len(puntos_data) < 3:
        print("No hay al menos tres sensores validos para interpolar.")
        return
    try:
        tri = Delaunay(puntos_data)
    except Exception as error:
        print(f"No se pudo construir la triangulacion Delaunay: {error}")
        return

    # Este poligono delimita estrictamente el area que los sensores pueden interpolar.
    area_interpolable = MultiPoint(puntos_data).convex_hull
    features = []
    for colonia in colonias_data:
        recorte = solo_poligonos(colonia["geometry"].intersection(area_interpolable))
        if recorte is None or recorte.is_empty:
            continue

        valores_en_colonia = [
            valores_puntos[i] for i, (lon, lat) in enumerate(puntos_data)
            if recorte.covers(Point(lon, lat))
        ]
        if valores_en_colonia:
            valor = float(np.mean(valores_en_colonia))
        else:
            punto = recorte.representative_point()
            indice = tri.find_simplex([[punto.x, punto.y]])[0]
            valor = (interpolar_lineal(np.array([punto.x, punto.y]), tri.simplices[indice],
                                       puntos_data, valores_puntos)
                     if indice >= 0 else None)

        valor = valor_numerico(valor)
        features.append({
            "type": "Feature",
            "geometry": mapping(recorte),  # conserva huecos y MultiPolygon para ArcGIS
            "properties": {
                "nombre": colonia["nombre"],
                "valor_interp": round(valor, 2) if valor is not None else None,
                "AQ": (clasificar_calidad_aire_pm25(valor) if contaminante == "pm2_5"
                       else clasificar_calidad_aire_pm10(valor)),
                "contaminante": contaminante,
                "timestamp_utc": timestamp,
            },
        })

    with open(nombre_archivo, "w", encoding="utf-8") as archivo:
        json.dump({"type": "FeatureCollection", "features": features}, archivo,
                  ensure_ascii=False, indent=2, allow_nan=False)
    print(f"GeoJSON recortado al area interpolable generado: {nombre_archivo}")


if __name__ == "__main__":
    timestamp_ejecucion = datetime.now(timezone.utc).isoformat()
    df_sensores = leer_csv(CSV_FILE)
    puntos_data, valores_pm25, valores_pm10 = crear_geojson(df_sensores, timestamp_ejecucion)

    if not os.path.exists(ARCHIVO_SHP_COLONIAS):
        print(f"No se encontro el shapefile: {ARCHIVO_SHP_COLONIAS}")
    elif len(puntos_data) >= 3:
        generar_geojson_colonias(SALIDA_GEOJSON_COLONIAS_PM25,
                                 cargar_datos_colonias_shp(ARCHIVO_SHP_COLONIAS),
                                 puntos_data, valores_pm25, "pm2_5", timestamp_ejecucion)
        generar_geojson_colonias(SALIDA_GEOJSON_COLONIAS_PM10,
                                 cargar_datos_colonias_shp(ARCHIVO_SHP_COLONIAS),
                                 puntos_data, valores_pm10, "pm10", timestamp_ejecucion)
    else:
        print("No se generan GeoJSON de colonias: se requieren tres sensores validos no colineales.")

    print("Proceso completado.")

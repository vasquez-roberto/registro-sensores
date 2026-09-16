import json
import os
from datetime import datetime, timezone
from pathlib import Path

import geojson
import numpy as np
import pandas as pd
import requests
import shapefile
from scipy.spatial import Delaunay, KDTree
from shapely.geometry import Point, mapping, shape

print("INICIANDO SCRIPT...")


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

SENSORES_BLOQUEADOS = {121825}

MIN_VALOR_PM = 0.1
MAX_VALOR_PM25 = 500.0
MAX_VALOR_PM10 = 1000.0


def leer_csv(ruta):
    df = pd.read_csv(ruta)
    df = df.dropna(subset=["latitude", "longitude", "sensor_index"])
    df["sensor_index"] = df["sensor_index"].astype(int)

    df = df[~df["sensor_index"].isin(SENSORES_BLOQUEADOS)].copy()

    return df


def consultar_sensores_masivo(sensor_indices):
    sensor_indices = [
        int(sensor_id)
        for sensor_id in sensor_indices
        if int(sensor_id) not in SENSORES_BLOQUEADOS
    ]

    if not sensor_indices:
        return {}

    if not API_KEY:
        print("Error: falta API_KEY_PURPLEAIR en el archivo .env.")
        return {}

    ids = ",".join(map(str, sensor_indices))
    url = (
        "https://api.purpleair.com/v1/sensors"
        f"?fields=pm1.0_atm,pm2.5_atm&show_only={ids}"
    )
    headers = {"X-API-Key": API_KEY}

    lecturas = {}

    try:
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code != 200:
            print(
                f"Error en PurpleAir: "
                f"{response.status_code} -> {response.text}"
            )
            return lecturas

        data = response.json()
        campos = data.get("fields", [])

        idx_id = campos.index("sensor_index")
        idx_pm1 = campos.index("pm1.0_atm")
        idx_pm25 = campos.index("pm2.5_atm")

        for sensor in data.get("data", []):
            sensor_id = int(sensor[idx_id])

            if sensor_id in SENSORES_BLOQUEADOS:
                continue

            lecturas[sensor_id] = (sensor[idx_pm1], sensor[idx_pm25])

    except Exception as error:
        print(f"Error consultando API de PurpleAir: {error}")

    return lecturas


def clasificar_calidad_aire_pm25(pm25):
    if pm25 is None or (isinstance(pm25, float) and np.isnan(pm25)):
        return "Sin datos"

    try:
        pm25 = float(pm25)
    except (ValueError, TypeError):
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
    except (ValueError, TypeError):
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
    features = []
    puntos = []
    valores_pm25 = []
    valores_pm10 = []
    datos_historicos = []

    indices = df["sensor_index"].tolist()
    lecturas = consultar_sensores_masivo(indices)

    for _, fila in df.iterrows():
        sensor_id = int(fila["sensor_index"])

        if sensor_id in SENSORES_BLOQUEADOS:
            continue

        pm10, pm25 = lecturas.get(sensor_id, (None, None))

        if pm10 is None or pm25 is None:
            continue

        try:
            pm10 = float(pm10)
            pm25 = float(pm25)
        except (ValueError, TypeError):
            continue

        es_pm25_valido = MIN_VALOR_PM <= pm25 <= MAX_VALOR_PM25
        es_pm10_valido = MIN_VALOR_PM <= pm10 <= MAX_VALOR_PM10

        if not (es_pm25_valido and es_pm10_valido):
            print(
                f"Sensor {sensor_id} ignorado por datos atípicos: "
                f"PM2.5={pm25}, PM10={pm10}"
            )
            continue

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
        valores_pm25.append(pm25)
        valores_pm10.append(pm10)

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
        df_total["sensor_index"] = pd.to_numeric(
            df_total["sensor_index"],
            errors="coerce",
        )
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
        matriz = np.array([delta1, delta2]).T
        pesos = np.linalg.solve(matriz, delta_p)

        return (
            (1 - pesos[0] - pesos[1]) * z0
            + pesos[0] * z1
            + pesos[1] * z2
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
        kdtree = KDTree(puntos_data)
    except Exception as error:
        print(f"Error al inicializar Delaunay / KDTree: {error}")
        tri = None
        kdtree = None

    geo_json_features = []

    for colonia in colonias_data:
        geom = colonia["geometry"]

        if not geom.is_valid or geom.is_empty:
            continue

        valores_en_colonia = [
            valores_puntos[i]
            for i, (lon, lat) in enumerate(puntos_data)
            if geom.contains(Point(lon, lat))
        ]

        if valores_en_colonia:
            val_interpolado = float(np.mean(valores_en_colonia))

        elif tri is None:
            val_interpolado = np.nan

        else:
            punto_interior = geom.representative_point()
            punto = np.array([punto_interior.x, punto_interior.y])

            indice_triangulo = tri.find_simplex(punto)

            if indice_triangulo != -1:
                val_interpolado = interpolar_lineal(
                    punto,
                    tri.simplices[indice_triangulo],
                    puntos_data,
                    valores_puntos,
                )
            else:
                _, indice_cercano = kdtree.query(punto)
                val_interpolado = valores_puntos[indice_cercano]

        valor_exportado = (
            round(float(val_interpolado), 2)
            if val_interpolado is not None and not np.isnan(val_interpolado)
            else None
        )

        geo_json_features.append(
            {
                "type": "Feature",
                "geometry": mapping(geom),
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

    geo_json_data = {
        "type": "FeatureCollection",
        "metadata": {"ultima_ejecucion_utc": timestamp},
        "features": geo_json_features,
    }

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
        colonias_base = cargar_datos_colonias_shp(ARCHIVO_SHP_COLONIAS)

        generar_geojson_colonias(
            SALIDA_GEOJSON_COLONIAS_PM25,
            colonias_base,
            puntos_data,
            valores_pm25,
            "pm2_5",
            timestamp_ejecucion,
        )

        generar_geojson_colonias(
            SALIDA_GEOJSON_COLONIAS_PM10,
            colonias_base,
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
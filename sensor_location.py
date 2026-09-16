import json
import os
import geojson
import numpy as np
import pandas as pd
import requests
import shapefile
from datetime import datetime, timezone
from scipy.spatial import Delaunay, KDTree
from shapely.geometry import Point, mapping, shape

print("INICIANDO SCRIPT...")

# --- CONFIGURACIÓN Y CONSTANTES ---
API_KEY = os.getenv(
    "API_KEY_PURPLEAIR", "C7B070E7-AAFE-11F1-9E30-4201AC1DC129"
)
CSV_FILE = "sensores_detectados.csv"
SALIDA_GEOJSON_SENSORES = "sensores.geojson"
SALIDA_GEOJSON_COLONIAS_PM25 = "AQ_PM25.geojson"
SALIDA_GEOJSON_COLONIAS_PM10 = "AQ_PM10.geojson"
ARCHIVO_SHP_COLONIAS = "shp/2025_1_19_A.shp"

# LISTA NEGRA: Agrega aquí los IDs (sensor_index) que desees bloquear
SENSORES_BLOQUEADOS = [121825]  # <--- REEMPLAZA CON LOS IDS REALES

# RANGOS DE VALIDEZ PARA CALIDAD DEL AIRE (µg/m³)
MIN_VALOR_PM = 0.1
MAX_VALOR_PM25 = 500.0
MAX_VALOR_PM10 = 1000.0


def leer_csv(ruta):
    df = pd.read_csv(ruta)
    df = df.dropna(subset=["latitude", "longitude", "sensor_index"])
    df["sensor_index"] = df["sensor_index"].astype(int)

    # --- FILTRO DE BLOQUEO DE SENSORES ---
    sensores_iniciales = len(df)
    df = df[~df["sensor_index"].isin(SENSORES_BLOQUEADOS)]
    sensores_excluidos = sensores_iniciales - len(df)

    if sensores_excluidos > 0:
        print(f"🚫 {sensores_excluidos} sensor(es) fueron bloqueados y excluidos del proceso.")
    # -------------------------------------

    return df


def consultar_sensores_masivo(sensor_indices):
    if not sensor_indices:
        return {}

    ids = ",".join(map(str, sensor_indices))
    url = f"https://api.purpleair.com/v1/sensors?fields=pm1.0_atm,pm2.5_atm&show_only={ids}"
    headers = {"X-API-Key": API_KEY} if API_KEY else {}

    lecturas = {}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            campos = data.get("fields", [])
            idx_id = campos.index("sensor_index")
            idx_pm1 = campos.index("pm1.0_atm")
            idx_pm25 = campos.index("pm2.5_atm")

            for s in data.get("data", []):
                lecturas[s[idx_id]] = (s[idx_pm1], s[idx_pm25])
    except Exception as e:
        print(f"Error consultando API de PurpleAir: {e}")
    return lecturas


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

    indices = df["sensor_index"].tolist()
    lecturas = consultar_sensores_masivo(indices)

    for _, fila in df.iterrows():
        s_id = int(fila["sensor_index"])
        pm10, pm25 = lecturas.get(s_id, (None, None))

        if pm10 is None or pm25 is None:
            continue

        try:
            pm10, pm25 = float(pm10), float(pm25)
        except (ValueError, TypeError):
            continue

        # --- FILTRO ESTRICTO DE ATÍPICOS (OUTLIERS) ---
        es_pm25_valido = MIN_VALOR_PM <= pm25 <= MAX_VALOR_PM25
        es_pm10_valido = MIN_VALOR_PM <= pm10 <= MAX_VALOR_PM10

        if not (es_pm25_valido and es_pm10_valido):
            print(
                f"⚠️ Sensor {s_id} ignorado por datos atípicos: PM2.5={pm25}, PM10={pm10}"
            )
            continue
        # ---------------------------------------------

        props = {
            "sensor_index": s_id,
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
        valores_pm25.append(pm25)
        valores_pm10.append(pm10)

        datos_historicos.append({
            "sensor_index": s_id,
            "name": fila.get("name", ""),
            "timestamp": timestamp,
            "pm1_0": pm10,
            "pm2_5": pm25,
        })

    with open(SALIDA_GEOJSON_SENSORES, "w", encoding="utf-8") as f:
        geojson.dump(geojson.FeatureCollection(features), f, indent=2)
    print(f"✓ GeoJSON de sensores generado: {SALIDA_GEOJSON_SENSORES}")

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
        kdtree = KDTree(puntos_data)
    except Exception as e:
        print(f"Error al inicializar Delaunay / KDTree: {e}")
        tri, kdtree = None, None

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
        else:
            if tri is None:
                val_interpolado = np.nan
            else:
                punto_interior = geom.representative_point()
                p_cent = np.array([punto_interior.x, punto_interior.y])

                idx = tri.find_simplex(p_cent)
                if idx != -1:
                    val_interpolado = interpolar_lineal(
                        p_cent, tri.simplices[idx], puntos_data, valores_puntos
                    )
                else:
                    _, idx_cercano = kdtree.query(p_cent)
                    val_interpolado = valores_puntos[idx_cercano]

        val_exp = (
            round(float(val_interpolado), 2)
            if val_interpolado is not None and not np.isnan(val_interpolado)
            else None
        )

        geo_json_features.append({
            "type": "Feature",
            "geometry": mapping(geom),
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

    geo_json_data = {
        "type": "FeatureCollection",
        "metadata": {"ultima_ejecucion_utc": timestamp},
        "features": geo_json_features,
    }

    with open(nombre_archivo, "w", encoding="utf-8") as f:
        json.dump(geo_json_data, f, ensure_ascii=False, indent=2)
    print(f"✓ GeoJSON generado: {nombre_archivo}")


if __name__ == "__main__":
    timestamp_ejecucion = datetime.now(timezone.utc).isoformat()
    df_sensores = leer_csv(CSV_FILE)
    puntos_data, valores_pm25, valores_pm10 = crear_geojson(
        df_sensores, timestamp_ejecucion
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
            df_hist, df_det = pd.read_csv("historico.csv"), pd.read_csv(
                CSV_FILE
            )
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
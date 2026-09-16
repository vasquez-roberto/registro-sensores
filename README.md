# Generación de capas de calidad del aire

## Problema identificado

Una lectura anómala podía alterar la interpolación. Además, el shapefile de AGEBs usa el sistema de coordenadas `MEXICO_ITRF_2008_LCC` (Lambert Conformal Conic, metros), mientras que los
sensores PurpleAir usan longitud/latitud. Un GeoJSON exportado directamente
desde ese SHP tenía coordenadas como `2659616, 1545244`; un GeoJSON para
ArcGIS debe emplear coordenadas geográficas, por ejemplo `-100.25, 25.70`.

## Solución aplicada

`sensor_location.py` guarda todos los sensores detectados. El
filtro ocurre después, en `script.py`, sobre los valores
medidos: por tanto no depende del ID de ningún sensor y funciona igual con
cualquier lectura atípica.

`script.py` abre el SHP con GeoPandas, toma su CRS del
archivo `.prj` y lo reproyecta a `EPSG:4326` antes de mezclarlo con los
sensores y escribir GeoJSON. Los archivos `AQ_PM25.geojson` y
`AQ_PM10.geojson` quedan listos para ArcGIS.

También se agrega un filtro de lecturas:

| Contaminante | Rango aceptado (µg/m³) |
| PM2.5 | 0 a 500.4 |
| PM1.0 | 0 a 604 |

Estos son límites de cribado, no pruebas de que una lectura superior sea
imposible: durante humo extremo puede haber concentraciones muy altas. Los
límites corresponden al extremo superior de las tablas de concentración del
AQI de EPA y sirven para impedir que valores claramente defectuosos entren a
la interpolación. EPA señala además que los datos PurpleAir requieren control
de calidad y corrección antes de compararse con monitores regulatorios.

El script ahora consulta `pm10.0` y `pm2.5`. La versión anterior consultaba
`pm1.0` pero lo etiquetaba como PM10; por eso sus resultados no eran PM10
real.

## Ejecución

El archivo `.env` debe estar junto al script e incluir:

```env
API_KEY_PURPLEAIR=tu_clave
```

Ejecuta primero el localizador y luego el generador:

```powershell
python .\sensor_location_automatico.py
python .\script_reproyectado_filtrado.py
```

## Cómo consultar las coordenadas del SHP

El CRS declarado por el SHP se obtiene desde su archivo `.prj`:

```powershell
Get-Content .\shp\2025_1_19_A.prj -Raw
```

Para confirmar los límites de las geometrías una vez reproyectadas:

```powershell
@'
import geopandas as gpd
gdf = gpd.read_file("shp/2025_1_19_A.shp").to_crs("EPSG:4326")
print("CRS:", gdf.crs)
print("Límites [min lon, min lat, max lon, max lat]:", gdf.total_bounds)
'@ | python
```

En Monterrey, las coordenadas reproyectadas deben estar aproximadamente cerca
de longitud `-100` y latitud `25`.

## Referencias

- [EPA AQI Breakpoints](https://aqs.epa.gov/aqsweb/documents/codetables/aqi_breakpoints.html)
- [EPA: Understanding Air Sensor Data](https://www.epa.gov/air-sensor-toolbox/understanding-air-sensor-data)
- [EPA: Quality Assurance for Air Sensors](
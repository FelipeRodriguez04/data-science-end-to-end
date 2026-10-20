from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

carpeta = Path(__file__).resolve().parent
figuras = carpeta.parent / "data-contenedor"
figuras.mkdir(exist_ok=True)

datos = pd.read_csv(
    carpeta / "obt_emergencias_canton_dia.csv",
    dtype={"cod_canton": "string", "cod_provincia": "string"},
    parse_dates=["fecha"],
)

# Exploramos el volumen futuro a tres días, insumo del objetivo del proyecto.
# La etiqueta de demanda alta (> P90 por cantón y día de semana) se construye
# después de separar train/test, ajustando el umbral solo con entrenamiento.
OBJETIVO = "obj_n_emergencias_t3"
PREDICTORA = "media_prev_7d"
print(f"Objetivo exploratorio: emergencias en t+3 ({OBJETIVO})")

# Paso 1: resumen.
print("Filas y columnas:", datos.shape)
print(datos.dtypes)
print(datos.isna().sum())
print(datos.describe().round(2))
print("Claves cantón-fecha duplicadas:", datos.duplicated(["cod_canton", "fecha"]).sum())

# Los últimos tres días de cada cantón todavía no tienen un futuro observado.
# Se excluyen del análisis del objetivo, sin reemplazarlos con ceros.
con_objetivo = datos.loc[datos[OBJETIVO].notna()].copy()

# Paso 2: distribucion de la variable objetivo.
plt.figure()
plt.hist(con_objetivo[OBJETIVO], bins=40)
plt.title("Distribución de emergencias en t+3")
plt.xlabel("Número de emergencias por cantón y día")
plt.ylabel("Cantidad de observaciones")
plt.yscale("log")
plt.tight_layout()
plt.savefig(figuras / "proyecto_distribucion_objetivo.png")
plt.close()

# Paso 3: atipicos de la variable objetivo con la regla del IQR.
q1 = con_objetivo[OBJETIVO].quantile(0.25)
q3 = con_objetivo[OBJETIVO].quantile(0.75)
iqr = q3 - q1
fuera = con_objetivo.loc[
    (con_objetivo[OBJETIVO] < q1 - 1.5 * iqr)
    | (con_objetivo[OBJETIVO] > q3 + 1.5 * iqr)
]
print(f"Límites IQR: {q1 - 1.5 * iqr:.2f} a {q3 + 1.5 * iqr:.2f}")
print(f"Atipicos de {OBJETIVO} segun IQR: {len(fuera)}")
print(fuera.head(10))

# Paso 4: el promedio de los siete días anteriores está disponible al cierre
# de t y puede ayudar a anticipar el volumen en t+3.
pares = con_objetivo.dropna(subset=[PREDICTORA])
plt.figure(figsize=(9, 6))
plt.scatter(pares[PREDICTORA], pares[OBJETIVO], s=4, alpha=0.15)
plt.title("Emergencias en t+3 frente al promedio de los siete días anteriores")
plt.xlabel("Promedio de emergencias de t-7 a t-1")
plt.ylabel("Emergencias en t+3")
plt.tight_layout()
plt.savefig(figuras / "proyecto_media_previa_vs_objetivo.png", dpi=150)
plt.close()
print("\nCorrelaciones con el volumen en t+3:")
variables = ["n_emergencias", "lag_1d", "lag_7d", "media_prev_7d", "media_prev_28d"]
print(datos[variables + [OBJETIVO]].corr()[OBJETIVO].loc[variables].round(4))
# La correlación conjunta puede reflejar diferencias de tamaño entre cantones;
# no mide por sí sola la capacidad de anticipar picos dentro de cada cantón.

# Paso 5: tres hallazgos que cambian cómo preparar el modelo.
# Consultas de respaldo: groupby y correlación para el hallazgo 1;
# filtro de faltantes para el hallazgo 2; filtro y groupby para el hallazgo 3.
print("\nHallazgo 1: los cantones tienen escalas de demanda diferentes.")
resumen_canton = con_objetivo.groupby(["cod_canton", "canton"])[OBJETIVO].agg(
    promedio="mean", maximo="max", observaciones="size"
)
print(resumen_canton.sort_values("promedio", ascending=False).head(10).round(2))
print("Rango de promedios por cantón:", resumen_canton["promedio"].agg(["min", "max"]).round(2))
# Comparamos la relación global con la relación al descontar la media de
# cada cantón: una correlación global alta puede reflejar su escala de demanda.
centrados = pares[[PREDICTORA, OBJETIVO]] - pares.groupby("cod_canton")[[PREDICTORA, OBJETIVO]].transform("mean")
print("Correlación global promedio previo / objetivo:", round(pares[PREDICTORA].corr(pares[OBJETIVO]), 4))
print("Correlación descontando la media por cantón:", round(centrados[PREDICTORA].corr(centrados[OBJETIVO]), 4))
print("Cantones con al menos un atípico global:", fuera["cod_canton"].nunique())
print("No eliminar estos valores por IQR global: revisar el cantón y definir su umbral con train.")

print("\nHallazgo 2: los faltantes temporales tienen una causa estructural.")
temporales = ["lag_1d", "lag_7d", "lag_14d", "lag_28d", "media_prev_7d",
              "media_prev_28d", "obj_n_emergencias_t3", "obj_n_emergencias_t7"]
print(datos[temporales].isna().sum())
# Verificamos si los nulos coinciden exactamente con los bordes de cada serie.
ordenados = datos.sort_values(["cod_canton", "fecha"])
posicion = ordenados.groupby("cod_canton").cumcount()
tamano = ordenados.groupby("cod_canton")["fecha"].transform("size")
for columna, ventana in {
    "lag_1d": 1, "lag_7d": 7, "lag_14d": 14, "lag_28d": 28,
    "media_prev_7d": 7, "media_prev_28d": 28,
    "obj_n_emergencias_t3": -3, "obj_n_emergencias_t7": -7,
}.items():
    esperado = posicion < ventana if ventana > 0 else posicion >= tamano + ventana
    faltantes = ordenados.loc[ordenados[columna].isna(), ["cod_canton", "fecha"]]
    discrepancias = (ordenados[columna].isna() != esperado).sum()
    print(f"{columna}: {len(faltantes)} faltantes; {discrepancias} discrepancias con los bordes esperados")
print("Filas con objetivo y promedio previo disponibles:", len(pares))
print("Respetar el calentamiento de las ventanas y el final sin etiqueta; no rellenar con cero.")

print("\nHallazgo 3: enero de 2024 está marcado como período anómalo por el pipeline.")
anomalos = datos.loc[datos["es_periodo_anomalo"], ["periodo", "cod_canton", "n_emergencias"]]
print("Períodos marcados y cantidad de filas:")
print(anomalos.groupby("periodo").size())
mensual = datos.groupby("periodo").agg(
    total_emergencias=("n_emergencias", "sum"),
    promedio_canton_dia=("n_emergencias", "mean"),
    periodo_anomalo=("es_periodo_anomalo", "max"),
)
print(mensual.loc[["2023-12", "2024-01", "2024-02"]].round(2))
print("Investigar su cobertura antes de entrenar; la bandera no demuestra por sí sola un error.")
print("Usar una separación temporal para evaluar. Las columnas obj_* son resultados futuros:")
print("ninguna debe entrar como predictora. El P90 de demanda alta se calcula solo con train.")
print(f"\nGráficos guardados en: {figuras}")

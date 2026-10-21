from pathlib import Path

import pandas as pd

ruta_datos = (
    Path(__file__).resolve().parents[2]
    / "semana-08-sesion-02"
    / "plantilla-eda"
    / "obt_emergencias_canton_dia.csv"
)
datos = pd.read_csv(
    ruta_datos,
    dtype={"cod_canton": "string", "cod_provincia": "string"},
    parse_dates=["fecha"],
)

# Momento de predicción: al cierre de t, anticipamos el volumen en t+3.
# Conservamos el objetivo exploratorio de la semana 8. Para clasificar demanda
# alta, el P90 por cantón y día de semana debe ajustarse solo con entrenamiento.
OBJETIVO = "obj_n_emergencias_t3"

# Feature 1: cambio relativo de la demanda reciente respecto a su nivel habitual.
# Hallazgo de la semana 8: los cantones tienen escalas de demanda diferentes.
# El cociente permite distinguir un aumento relativo de un simple volumen alto.
# Ambas medias usan días anteriores a t y están disponibles al predecir.
# Se conserva NaN durante el calentamiento o cuando la media de 28 días es cero:
# no tener historia suficiente no equivale a tener demanda cero.
base = datos["media_prev_28d"].where(datos["media_prev_28d"] > 0)
datos["demanda_relativa_7d_28d"] = datos["media_prev_7d"] / base

# Feature 2: el calendario del día objetivo puede capturar cambios semanales
# en movilidad y actividad. Es una hipótesis de dominio que evaluamos abajo.
# La fecha de t+3 se conoce al cierre de t, sin consultar emergencias futuras.
# Es binaria y se usa como 0/1 en regresión lineal y árboles; no requiere ordinal.
fecha_objetivo = datos["fecha"] + pd.Timedelta(days=3)
datos["es_fin_de_semana_t3"] = (fecha_objetivo.dt.dayofweek >= 5).astype(int)

print(f"Predicción al cierre de t: número de emergencias en t+3 ({OBJETIVO}).")

# Evidencia exploratoria: excluir finales sin etiqueta y ventanas incompletas
# solo para cada comparación; conservar todas las filas en datos.
pares = datos.dropna(subset=["demanda_relativa_7d_28d", OBJETIVO])
print("\nFeature 1: demanda relativa (1 = nivel habitual; >1 = aumento).")
print("Observaciones válidas:", len(pares))
print(pares[["demanda_relativa_7d_28d", OBJETIVO]].corr().round(4))
# Descontar la media por cantón ayuda a examinar cambios dentro de cada cantón.
columnas = ["demanda_relativa_7d_28d", OBJETIVO]
centrados = pares[columnas] - pares.groupby("cod_canton")[columnas].transform("mean")
print("Correlación descontando la media por cantón:")
print(centrados.corr().round(4))
# En este CSV las correlaciones son aproximadamente 0.0012 y 0.0100:
# no hay evidencia de una relación lineal fuerte con el volumen futuro.
# Es una candidata motivada por el hallazgo, no una mejora ya demostrada.

print("\nFeature 2: objetivo promedio según fin de semana en t+3 (0=no, 1=sí).")
con_objetivo = datos.dropna(subset=[OBJETIVO])
print(
    con_objetivo.groupby("es_fin_de_semana_t3")[OBJETIVO]
    .agg(promedio="mean", observaciones="count")
    .round(4)
)
# En este CSV: 38.31 emergencias en días laborables frente a 50.49 en fin
# de semana. La diferencia apoya evaluar esta feature, sin probar causalidad.

print("\nFaltantes en las nuevas features:")
print(datos[["demanda_relativa_7d_28d", "es_fin_de_semana_t3"]].isna().sum())
print("\nMuestra de features y objetivo:")
print(pares[["cod_canton", "fecha", *columnas, "es_fin_de_semana_t3"]].head())
print(
    "\nEstas asociaciones no demuestran mejora predictiva: comparar modelos con "
    "validación temporal. Ninguna columna obj_* debe entrar como predictora."
)

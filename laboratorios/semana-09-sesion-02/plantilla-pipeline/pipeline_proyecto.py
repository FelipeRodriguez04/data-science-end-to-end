from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Al cierre de t predecimos el volumen de emergencias por cantón en t+3.
# Se mantiene el objetivo exploratorio de la sesión anterior. Una etiqueta de
# demanda alta requeriría calcular su P90 exclusivamente con entrenamiento.
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

# Las mismas dos features de la sesión pasada, sin usar resultados futuros.
base = datos["media_prev_28d"].where(datos["media_prev_28d"] > 0)
datos["demanda_relativa_7d_28d"] = datos["media_prev_7d"] / base
fecha_objetivo = datos["fecha"] + pd.Timedelta(days=3)
datos["es_fin_de_semana_t3"] = (fecha_objetivo.dt.dayofweek >= 5).astype(int)

OBJETIVO = "obj_n_emergencias_t3"
# Los conteos de t están disponibles porque predecimos después del cierre.
NUMERICAS = [
    "n_emergencias", "lag_1d", "lag_7d", "lag_14d", "lag_28d",
    "media_prev_7d", "media_prev_28d", "demanda_relativa_7d_28d",
]
# El código de cantón se codifica como categoría, conservando sus ceros iniciales.
CATEGORICAS = ["cod_canton"]
YA_LISTAS = ["es_fin_de_semana_t3", "es_feriado_t3"]
# No se incluyen obj_*, identificadores de fila ni banderas retrospectivas
# como es_periodo_anomalo. Los feriados futuros se conocen por calendario.

X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS]
y = datos[OBJETIVO]

# El último 20% de fechas con objetivo observado se reserva para prueba.
# Se dividen fechas completas para mantener juntos todos los cantones de un día.
fechas = datos.loc[y.notna(), "fecha"].drop_duplicates().sort_values()
fecha_corte = fechas.iloc[int(len(fechas) * 0.8)]
# Además, las etiquetas de entrenamiento deben haber ocurrido antes del corte:
# excluir las últimas tres fechas previas evita usar resultados de prueba.
es_entrenamiento = y.notna() & (fecha_objetivo < fecha_corte)
es_prueba = y.notna() & (datos["fecha"] >= fecha_corte)
X_ent, y_ent = X.loc[es_entrenamiento], y.loc[es_entrenamiento]
X_prueba, y_prueba = X.loc[es_prueba], y.loc[es_prueba]

print("Objetivo:", OBJETIVO, "| Predicción al cierre de t para t+3")
print("Fecha de corte:", fecha_corte.date())
for nombre, mascara in [("Entrenamiento", es_entrenamiento), ("Prueba", es_prueba)]:
    fechas_grupo = datos.loc[mascara, "fecha"]
    print(f"{nombre}: {fechas_grupo.min().date()} a {fechas_grupo.max().date()}")
print("Filas sin objetivo excluidas:", int(y.isna().sum()))
print("Filas excluidas por separación de etiquetas:", int((
    y.notna() & (datos["fecha"] < fecha_corte) & ~es_entrenamiento
).sum()))

preparacion = ColumnTransformer([
    ("numericas", Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ]), NUMERICAS),
    ("categoricas", Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        # Matriz dispersa para no almacenar miles de ceros por cantón.
        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ]), CATEGORICAS),
    ("ya_listas", "passthrough", YA_LISTAS),
], sparse_threshold=1.0)

X_ent_listo = preparacion.fit_transform(X_ent)
X_prueba_listo = preparacion.transform(X_prueba)
print("Entrenamiento:", X_ent_listo.shape, "Prueba:", X_prueba_listo.shape)
print(list(preparacion.get_feature_names_out()))
print("\nMedianas aprendidas solo con entrenamiento:")
print(pd.Series(
    preparacion.named_transformers_["numericas"].named_steps["imputar"].statistics_,
    index=NUMERICAS,
))
# Los faltantes iniciales de las ventanas se imputan con medianas de train;
# los objetivos ausentes nunca se imputan. En prueba se usa solo transform.

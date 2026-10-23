from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ruta_datos = Path(__file__).resolve().parent.parent / "starter" / "viajes_features.csv"
viajes = pd.read_csv(ruta_datos)

NUMERICAS = ["distancia_km", "pasajeros"]
YA_LISTAS = ["hora_sin", "hora_cos", "es_fin_de_semana", "es_hora_pico", "es_tarifa_fija"]
CATEGORICAS = ["zona_origen", "zona_destino", "metodo_pago"]

X = viajes[NUMERICAS + YA_LISTAS + CATEGORICAS]
y = viajes["tarifa"]

# TODO 1: divide X e y en entrenamiento (80%) y prueba (20%), con
# random_state=42.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TODO 2: arma un Pipeline para las columnas NUMERICAS (imputar con la
# mediana, despues escalar) y otro para las CATEGORICAS (imputar con el
# valor mas frecuente, despues OneHotEncoder con handle_unknown="ignore" y
# sparse_output=False). Juntalos en un ColumnTransformer llamado
# "preparacion", con las columnas YA_LISTAS pasando sin cambios
# ("passthrough").
pipeline_numericas = Pipeline([
    ("imputador", SimpleImputer(strategy="median")),
    ("escalador", StandardScaler()),
])
pipeline_categoricas = Pipeline([
    ("imputador", SimpleImputer(strategy="most_frequent")),
    ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preparacion = ColumnTransformer([
    ("numericas", pipeline_numericas, NUMERICAS),
    ("categoricas", pipeline_categoricas, CATEGORICAS),
    ("ya_listas", "passthrough", YA_LISTAS),
])

# TODO 3: aprende la preparacion solo con entrenamiento y transforma
# entrenamiento y prueba. Imprime cuantas columnas quedan.
X_train_preparado = preparacion.fit_transform(X_train)
X_test_preparado = preparacion.transform(X_test)
columnas = preparacion.get_feature_names_out()
encoder = preparacion.named_transformers_["categoricas"].named_steps["one_hot"]
print("Forma de entrenamiento:", X_train_preparado.shape)
print("Forma de prueba:", X_test_preparado.shape)
print("Total de columnas:", len(columnas))
print(
    f"Desglose: {len(NUMERICAS)} numéricas + "
    f"{sum(len(categorias) for categorias in encoder.categories_)} one-hot + "
    f"{len(YA_LISTAS)} ya listas."
)
print("Nombres de las columnas:", columnas.tolist())

# TODO 4: imprime las medianas que aprendio el imputador de numericas
# (atributo statistics_) junto a las medianas de entrenamiento y las de
# todo el dataset.
imputador = preparacion.named_transformers_["numericas"].named_steps["imputador"]
comparacion = pd.DataFrame({
    "mediana_aprendida": imputador.statistics_,
    "mediana_train": X_train[NUMERICAS].median(),
    "mediana_dataset": X[NUMERICAS].median(),
}, index=NUMERICAS)
print("\nComparación de medianas:")
print(comparacion)
print(
    "Las medianas aprendidas proceden únicamente de entrenamiento. Que alguna "
    "coincida con la del dataset completo no significa que se usó prueba."
)

# TODO 5: transforma el viaje nuevo de abajo, que sale de una zona que el
# entrenamiento nunca vio, e imprime cuantas columnas de zona_origen
# quedan en 1.
nuevo = pd.DataFrame([{
    "distancia_km": 9.0, "pasajeros": 1, "hora_sin": 0.5, "hora_cos": -0.866,
    "es_fin_de_semana": 0, "es_hora_pico": 1, "es_tarifa_fija": 0,
    "zona_origen": "Bronx", "zona_destino": "Midtown", "metodo_pago": "tarjeta",
}])
nuevo_preparado = preparacion.transform(nuevo)
columnas_origen = [
    i for i, nombre in enumerate(columnas)
    if nombre.startswith("categoricas__zona_origen_")
]
print("\nForma del viaje nuevo:", nuevo_preparado.shape)
print("Zonas de origen aprendidas:", encoder.categories_[0].tolist())
print(
    "Columnas de zona_origen en 1 para Bronx:",
    int((nuevo_preparado[0, columnas_origen] == 1).sum()),
)
print(
    "Bronx es desconocido: handle_unknown='ignore' deja su bloque de origen "
    "en ceros y conserva las mismas columnas. pd.get_dummies aplicado por "
    "separado crearía zona_origen_Bronx y faltaría el resto de zonas de train, "
    "por lo que habría que alinear las columnas."
)

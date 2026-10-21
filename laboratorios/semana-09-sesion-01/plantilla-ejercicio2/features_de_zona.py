from pathlib import Path

import pandas as pd

ruta_datos = Path(__file__).resolve().parent.parent / "starter" / "viajes_taxi.csv"
viajes = pd.read_csv(ruta_datos)
columnas_originales = viajes.columns.tolist()
viajes = viajes[(viajes["tarifa"] != 999.99) & (viajes["duracion_min"] >= 0)].copy()

ZONAS_MANHATTAN = ["Midtown", "Upper East Side", "Upper West Side", "Lower Manhattan", "Harlem"]

# TODO 1: crea la columna es_tarifa_fija (1 o 0): vale 1 cuando el viaje
# va de JFK Airport a una zona de Manhattan, o de una zona de Manhattan a
# JFK Airport. Imprime cuantos viajes la tienen en 1, y la tarifa minima
# y maxima de esos viajes.
es_jfk_a_manhattan = (viajes["zona_origen"] == "JFK Airport") & viajes[
    "zona_destino"
].isin(ZONAS_MANHATTAN)
es_manhattan_a_jfk = viajes["zona_origen"].isin(ZONAS_MANHATTAN) & (
    viajes["zona_destino"] == "JFK Airport"
)
viajes["es_tarifa_fija"] = (es_jfk_a_manhattan | es_manhattan_a_jfk).astype(int)
tarifas_fijas = viajes.loc[viajes["es_tarifa_fija"] == 1, "tarifa"]

print("Viajes con tarifa fija:", len(tarifas_fijas))
print("Tarifa mínima:", tarifas_fijas.min())
print("Tarifa máxima:", tarifas_fijas.max())

# TODO 2: codifica zona_origen con one-hot (pd.get_dummies), con el
# prefijo "origen" y dtype=int. Imprime los nombres de las columnas
# nuevas.
origen_one_hot = pd.get_dummies(viajes["zona_origen"], prefix="origen", dtype=int)
viajes = pd.concat([viajes, origen_one_hot], axis=1)
print("Columnas one-hot:", origen_one_hot.columns.tolist())

# TODO 3: codifica zona_origen de forma ordinal: un numero entero por
# zona, en orden alfabetico, en una columna origen_ordinal.
zonas_ordenadas = sorted(viajes["zona_origen"].dropna().unique())
codigo_por_zona = {zona: codigo for codigo, zona in enumerate(zonas_ordenadas)}
viajes["origen_ordinal"] = viajes["zona_origen"].map(codigo_por_zona).astype(int)
print("Codificación ordinal:", codigo_por_zona)
# Los códigos identifican zonas: su orden no representa distancia ni jerarquía.

# TODO 4: para cada columna original del dataset, decide si existe en el
# momento en que alguien pide el taxi (antes del viaje) e imprime la
# decision.
# Se supone que al pedir el taxi se indican destino y número de pasajeros.
# Como en el ejercicio 1, se usa la hora de inicio como hora de la solicitud
# inmediata; si hay espera, se debe usar la hora real de solicitud.
# Distancia y duración son las medidas reales, no estimaciones de la ruta.
disponibilidad = {
    "id_viaje": (True, "Se asigna al registrar la solicitud; es un identificador, no una feature útil."),
    "fecha_hora_inicio": (True, "Se asume una solicitud inmediata y se usa su fecha y hora, como en el ejercicio 1."),
    "zona_origen": (True, "Se indica el punto de recogida al pedir el taxi."),
    "zona_destino": (True, "Se supone que el destino se indica al pedir el taxi."),
    "distancia_km": (False, "La distancia real recorrida se conoce al finalizar el viaje."),
    "duracion_min": (False, "La duración real se conoce al finalizar el viaje."),
    "pasajeros": (True, "Se supone que se declara el número de pasajeros al pedir el taxi."),
    "metodo_pago": (False, "No se garantiza que el método de pago final esté elegido al solicitar el taxi."),
    "tarifa": (False, "La tarifa final es la variable a predecir y se confirma al terminar."),
    "propina": (False, "La propina se decide al pagar, después del viaje."),
}

print("\nDisponibilidad al pedir el taxi:")
for columna in columnas_originales:
    disponible, motivo = disponibilidad[columna]
    print(f"{columna}: {'SÍ' if disponible else 'NO'}. {motivo}")

print("\nUsar información posterior al momento de predicción produce fuga de datos.")

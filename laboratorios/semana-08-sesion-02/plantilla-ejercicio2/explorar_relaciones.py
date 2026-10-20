from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

carpeta = Path(__file__).resolve().parent
figuras = carpeta / "figuras"
figuras.mkdir(exist_ok=True)
viajes = pd.read_csv(carpeta.parent / "starter" / "viajes_taxi.csv")

# 1. Quitar solo los errores confirmados; conservar los demás atípicos.
errores = viajes["tarifa"].eq(999.99) | viajes["duracion_min"].lt(0)
limpios = viajes.loc[~errores].copy()
print(f"Filas originales: {len(viajes)}")
print(f"Errores eliminados: {errores.sum()}")
print(f"Filas restantes: {len(limpios)}")

# 2. Explorar la relación entre distancia y tarifa.
plt.figure(figsize=(9, 6))
plt.scatter(limpios["distancia_km"], limpios["tarifa"], alpha=0.4, s=15)
plt.title("Distancia contra tarifa (sin errores confirmados)")
plt.xlabel("Distancia (km)")
plt.ylabel("Tarifa (USD)")
plt.tight_layout()
plt.savefig(figuras / "distancia_vs_tarifa.png", dpi=150)
plt.close()

# La banda horizontal de 70 USD sugiere una tarifa fija. Verificamos
# sus zonas para comprobar si corresponde a viajes de JFK.
tarifa_fija = limpios.loc[limpios["tarifa"].eq(70)]
print(f"\nViajes con tarifa de 70 USD: {len(tarifa_fija)}")
print("Zonas de los viajes con tarifa de 70 USD:")
print(tarifa_fija.groupby(["zona_origen", "zona_destino"]).size())

# La columna de puntos en x=0 también se aparta de la tendencia general.
# Se cuenta para investigarla; todavía no es un error confirmado.
distancia_cero = limpios.loc[
    limpios["distancia_km"].eq(0) & limpios["tarifa"].gt(0)
]
print(f"Viajes con distancia 0 y tarifa positiva: {len(distancia_cero)}")

# 3. Incluir también los viajes sin método de pago registrado.
propinas_por_pago = limpios.groupby("metodo_pago", dropna=False).agg(
    propina_promedio=("propina", "mean"),
    viajes_propina_cero=("propina", lambda propinas: propinas.eq(0).sum()),
    cantidad_viajes=("propina", "size"),
)
print("\nPropinas por método de pago (NaN = sin registro):")
print(propinas_por_pago)
# Los ceros en efectivo pueden reflejar cómo se registra la propina;
# estos datos por sí solos no prueban que esos pasajeros no den propina.

# 4. Seleccionar explícitamente las variables numéricas del ejercicio.
variables = ["distancia_km", "duracion_min", "pasajeros", "propina"]
correlaciones = limpios[variables + ["tarifa"]].corr()["tarifa"].loc[variables]
print("\nCorrelaciones con la tarifa:")
print(correlaciones)
# La duración real del viaje no está disponible al pedir el taxi.
# Usarla en ese momento para predecir la tarifa sería fuga de información.

# 5. Distinguir los faltantes de los ceros sin imputar ni eliminar filas.
print(f"\nViajes con pasajeros vacío: {limpios['pasajeros'].isna().sum()}")
print(f"Viajes con pasajeros en 0: {limpios['pasajeros'].eq(0).sum()}")

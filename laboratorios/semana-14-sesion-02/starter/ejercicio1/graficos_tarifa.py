"""Ejercicio 1: los resultados de noviembre, para dos audiencias.

tarifas_noviembre.csv tiene los 1156 viajes de noviembre del monitoreo de
la Semana 13: la tarifa que estimo la version 1 del modelo, la que se
cobro, y si el viaje tenia tarifa fija. Este script calcula el resumen por
semana y dibuja dos graficos con los mismos datos: uno para el equipo de
datos (grafico_tecnico.png) y otro para la gerencia (grafico_ejecutivo.png).

    python graficos_tarifa.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

MAE_PRUEBA = 1.17

viajes = pd.read_csv("tarifas_noviembre.csv")
viajes["semana"] = pd.to_datetime(viajes["fecha_hora_inicio"]).dt.to_period("W").dt.start_time
viajes["error"] = (viajes["tarifa_cobrada"] - viajes["tarifa_estimada"]).abs()
viajes["diferencia"] = viajes["tarifa_cobrada"] - viajes["tarifa_estimada"]
# En centavos, una tarifa exactamente 20 % mayor queda fuera del conteo.
centavos_cobrados = (100 * viajes["tarifa_cobrada"]).round()
centavos_estimados = (100 * viajes["tarifa_estimada"]).round()
viajes["pagan_20_mas"] = 5 * centavos_cobrados > 6 * centavos_estimados

medidas = viajes.groupby("semana").agg(
    viajes=("error", "size"),
    mae=("error", "mean"),
    pagan_20_mas=("pagan_20_mas", "mean"),
    diferencia=("diferencia", "mean"),
)
medidas.index = medidas.index.strftime("%d/%m").rename("semana")
resumen = medidas.round(3)
print(resumen.to_string())

por_tipo = viajes.groupby(["semana", "es_tarifa_fija"])["error"].mean().unstack()
por_tipo = por_tipo.rename(columns={0: "Taxímetro", 1: "Tarifa fija"})
por_tipo.index = por_tipo.index.strftime("%d/%m")
figura, eje = plt.subplots(figsize=(7.6, 4.2))
por_tipo.plot(kind="bar", ax=eje, color=["#206A89", "#B68127"], rot=0, width=0.72)
eje.axhline(MAE_PRUEBA, color="#363E47", linestyle="--", label="MAE global de prueba (1,17 USD)")
eje.set(title="Error absoluto medio por semana y tipo de tarifa", xlabel="Inicio de semana", ylabel="MAE (USD por viaje)")
eje.grid(axis="y", color="#E1E7ED", linewidth=0.8)
eje.set_axisbelow(True)
eje.legend(frameon=False, loc="upper left")
figura.tight_layout()
figura.savefig("grafico_tecnico.png", dpi=150)
plt.close(figura)

figura, eje = plt.subplots(figsize=(7.6, 4.2))
porcentajes = medidas["pagan_20_mas"] * 100
colores = ["#9DAEBB"] * len(porcentajes)
colores[-2:] = ["#A33D27"] * min(len(porcentajes), 2)
barras = eje.bar(porcentajes.index, porcentajes, color=colores, width=0.55)
etiquetas = [f"{valor:.1f}".replace(".", ",") + " %" for valor in porcentajes]
eje.bar_label(barras, labels=etiquetas, padding=5, fontweight="bold")
eje.set_ylim(0, max(10, porcentajes.max() * 1.20))
eje.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
eje.set(
    title="Desde mediados de noviembre, más viajes superaron el precio estimado",
    xlabel="Inicio de semana",
    ylabel="Viajes con cobro más de 20 %\npor encima de lo estimado",
)
eje.spines[["top", "right"]].set_visible(False)
figura.tight_layout()
figura.savefig("grafico_ejecutivo.png", dpi=150)
plt.close(figura)
print("Graficos guardados: grafico_tecnico.png y grafico_ejecutivo.png")

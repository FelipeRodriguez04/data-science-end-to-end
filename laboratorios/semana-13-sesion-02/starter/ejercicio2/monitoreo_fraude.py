"""Ejercicio 2: el monitoreo del servicio de fraude.

El servicio de fraude (version 2, el gradient boosting) guarda cada pago que
evalua en predicciones_fraude.csv: cuando llego, la probabilidad, si lo mando
a revision y cuanto tardo en responder. Los fraudes no se conocen en el
momento: se confirman cuando el titular de la tarjeta reclama el cobro (un
contracargo), entre 2 y 21 dias despues, y quedan en contracargos.csv.

Hoy es 14 de diciembre. El reporte separa las metricas del sistema (la
latencia), una metrica que se puede calcular el mismo dia (la tasa de
alertas) y la metrica del modelo que le importa al negocio (el recall), que
solo se puede calcular para las semanas cuyos fraudes ya se confirmaron.

    python monitoreo_fraude.py

Despues de resolver, corre python comprobar.py hasta que los cinco pasos
digan OK.
"""

import pandas as pd

HOY = "2026-12-14"
DEMORA_MAXIMA = 21          # dias que puede tardar en confirmarse un fraude
TASA_REFERENCIA = 0.0758    # pagos enviados a revision en la prueba, con el umbral 0.10
RECALL_REFERENCIA = 0.661   # recall en la prueba, con el umbral 0.10
LATENCIA_MAXIMA_MS = 50


def con_dia_y_semana(predicciones):
    """Agrega "dia" (la fecha) y "semana" (el lunes de la semana) a cada pago."""
    predicciones = predicciones.copy()
    fecha = pd.to_datetime(predicciones["fecha_hora"])
    predicciones["dia"] = fecha.dt.date
    predicciones["semana"] = fecha.dt.to_period("W").dt.start_time.dt.date
    return predicciones


def latencia_p95(predicciones):
    """La latencia p95 de cada dia, en milisegundos: una Series indexada por dia."""
    return predicciones.groupby("dia")["latencia_ms"].quantile(0.95)


def tasa_de_alertas(predicciones):
    """Fracción semanal de pagos enviados a revisión."""
    return predicciones.groupby("semana")["revisar"].mean()


def fraudes_confirmados(contracargos, hoy):
    """IDs de los fraudes confirmados hasta el corte inclusive."""
    return set(contracargos.loc[contracargos["fecha_confirmacion"] <= hoy, "id_pago"])


def recall_por_semana(predicciones, contracargos, hoy):
    """Cuenta fraudes y detecciones solo en semanas maduras."""
    corte = pd.Timestamp(hoy) - pd.Timedelta(days=6 + DEMORA_MAXIMA)
    maduras = predicciones.loc[predicciones["semana"] <= corte.date()].copy()
    maduras["fraude"] = maduras["id_pago"].isin(fraudes_confirmados(contracargos, hoy))
    maduras["detectado"] = maduras["fraude"] & maduras["revisar"].eq(1)
    tabla = maduras.groupby("semana").agg(fraudes=("fraude", "sum"), detectados=("detectado", "sum"))
    tabla["recall"] = (tabla["detectados"] / tabla["fraudes"]).round(3)
    return tabla


def alertas(predicciones, contracargos, hoy):
    """Lista de alertas de latencia, tasa de revisión y recall."""
    observadas = predicciones.loc[predicciones["dia"] <= pd.Timestamp(hoy).date()]
    mensajes = []
    for dia, p95 in latencia_p95(observadas).items():
        if p95 > LATENCIA_MAXIMA_MS:
            mensajes.append(f"sistema: el {dia} la latencia p95 fue {p95:.1f} ms")
    for semana, tasa in tasa_de_alertas(observadas).items():
        if tasa < 0.75 * TASA_REFERENCIA:
            mensajes.append(f"modelo (indirecta): la semana del {semana} mando a revision {tasa:.3f} de los pagos")
    for semana, fila in recall_por_semana(observadas, contracargos, hoy).iterrows():
        if fila["fraudes"] and fila["detectados"] / fila["fraudes"] < 0.75 * RECALL_REFERENCIA:
            mensajes.append(f"modelo: la semana del {semana} detecto {fila['recall']:.3f} de los fraudes")
    return mensajes


if __name__ == "__main__":
    predicciones = con_dia_y_semana(pd.read_csv("predicciones_fraude.csv"))
    predicciones = predicciones.loc[predicciones["dia"] <= pd.Timestamp(HOY).date()]
    contracargos = pd.read_csv("contracargos.csv")
    print(f"{len(predicciones)} pagos evaluados del {predicciones['dia'].min()} al {predicciones['dia'].max()}")
    print("\nLatencia p95 por dia (ms), los cinco dias mas lentos:")
    print(latencia_p95(predicciones).sort_values(ascending=False).head(5).round(1).to_string())
    print("\nTasa de alertas por semana:")
    print(tasa_de_alertas(predicciones).round(4).to_string())
    print(f"\nRecall por semana madura, al {HOY}:")
    print(recall_por_semana(predicciones, contracargos, HOY).to_string())
    print("\nAlertas:")
    for mensaje in alertas(predicciones, contracargos, HOY):
        print("-", mensaje)

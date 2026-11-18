"""Evalúa pronósticos t+3 por semana objetivo y versión con observaciones cerradas."""
import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

CARPETA = Path(__file__).resolve().parent
CLAVES = ["cod_canton", "fecha_objetivo"]
COLUMNAS = [
    "semana_objetivo", "version_modelo", "pronosticos_vencidos",
    "evaluados", "sin_verdad", "mae", "costo_asimetrico",
]


def reporte(registro, verdad, hasta):
    filas = []
    with registro.open(encoding="utf-8") as archivo:
        for linea in archivo:
            if linea.strip():
                evento = json.loads(linea)
                filas.append(evento["salida"])
    if not filas:
        return pd.DataFrame(columns=COLUMNAS)

    pronosticos = pd.DataFrame(filas)
    pronosticos["fecha_objetivo"] = pd.to_datetime(pronosticos["fecha_objetivo"])
    pronosticos = pronosticos.loc[pronosticos["fecha_objetivo"] <= pd.Timestamp(hasta)].copy()
    pronosticos["semana_objetivo"] = pronosticos["fecha_objetivo"].dt.to_period("W-SUN").dt.start_time

    observaciones = pd.read_csv(
        verdad, dtype={"cod_canton": "string"}, usecols=["cod_canton", "fecha", "n_emergencias"],
        parse_dates=["fecha"],
    ).rename(columns={"fecha": "fecha_objetivo", "n_emergencias": "real"})
    unidos = pronosticos.merge(observaciones, on=CLAVES, how="left", validate="many_to_one")
    error = unidos["real"] - unidos["emergencias_estimadas"]
    unidos["error_absoluto"] = error.abs()
    unidos["costo"] = 3 * error.clip(lower=0) + (-error).clip(lower=0)
    tabla = unidos.groupby(["semana_objetivo", "version_modelo"], dropna=False).agg(
        pronosticos_vencidos=("real", "size"), evaluados=("real", "count"),
        mae=("error_absoluto", "mean"), costo_asimetrico=("costo", "mean"),
    ).reset_index()
    tabla["sin_verdad"] = tabla["pronosticos_vencidos"] - tabla["evaluados"]
    return tabla[COLUMNAS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registro", type=Path, default=CARPETA / "predicciones.jsonl")
    parser.add_argument("--verdad", type=Path, required=True)
    parser.add_argument("--hasta", type=date.fromisoformat, required=True,
                        help="Último día que debió estar cerrado; formato AAAA-MM-DD")
    args = parser.parse_args()
    tabla = reporte(args.registro, args.verdad, args.hasta)
    if tabla.empty:
        print("No hay pronósticos vencidos para evaluar.")
        return
    print(tabla.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

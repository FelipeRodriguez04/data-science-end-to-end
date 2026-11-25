"""Calibra el PSI con ventanas diarias de la prueba temporal de ECU911."""

import argparse
import hashlib
import json
from math import ceil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modelado.pipeline_proyecto import RUTA_DATOS, cargar_y_dividir
from servicio.deriva import FEATURES, medir_psi, proporciones

SERVICIO = Path(__file__).resolve().parents[1] / "servicio"


def medir(filas, referencia, observados):
    return {
        **medir_psi(filas, referencia),
        "mae": np.abs(observados - filas["prediccion"]).mean(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, default=RUTA_DATOS)
    parser.add_argument("--modelo", type=Path, default=SERVICIO / "modelo.joblib")
    parser.add_argument("--salida", type=Path, default=SERVICIO / "referencia_deriva.json")
    args = parser.parse_args()

    X_ent, X_prueba, _, y_prueba = cargar_y_dividir(args.datos)
    fechas = pd.read_csv(args.datos, usecols=["fecha"], parse_dates=["fecha"])["fecha"]
    prueba = X_prueba.assign(fecha=fechas, observado=y_prueba)
    modelo = joblib.load(args.modelo)
    prueba["prediccion"] = modelo.predict(X_prueba)

    referencia = {}
    for feature in FEATURES:
        serie = X_ent[feature]
        cortes = np.unique(serie.quantile(np.linspace(0.1, 0.9, 9))).tolist()
        referencia[feature] = {"cortes": cortes, "proporciones": proporciones(serie, cortes).tolist()}

    por_canton = X_ent.groupby("cod_canton")["n_emergencias"].mean()
    cantones_alta_demanda = por_canton[por_canton >= por_canton.quantile(0.90)].index
    lotes = prueba.groupby("fecha", sort=True)
    ventana = int(lotes.size().mode().iloc[-1])
    normales, datos, conceptual = [], [], []
    for semilla, (_, lote) in enumerate(lotes):
        if len(lote) != ventana:
            continue
        normales.append(medir(lote, referencia, lote["observado"]))
        pesos = np.where(lote["cod_canton"].isin(cantones_alta_demanda), 4, 1)
        sesgado = lote.sample(ventana, replace=True, weights=pesos, random_state=semilla)
        datos.append(medir(sesgado, referencia, sesgado["observado"]))
        conceptual.append(medir(lote, referencia, (1.20 * lote["observado"]).round()))

    escenarios = {
        "Normal": pd.DataFrame(normales),
        "Datos (alta demanda ×4)": pd.DataFrame(datos),
        "Conceptual (conteo +20%)": pd.DataFrame(conceptual),
    }
    maximo_normal = escenarios["Normal"][list(FEATURES)].max().max()
    umbral = ceil(100 * maximo_normal) / 100
    print(f"Modelo SHA-256: {hashlib.sha256(args.modelo.read_bytes()).hexdigest()}")
    print(f"Prueba del {prueba.fecha.min().date()} al {prueba.fecha.max().date()}")
    print(f"{ventana} predicciones por ventana, {len(normales)} ventanas por escenario")
    print(f"{len(cantones_alta_demanda)} cantones de alta demanda; umbral PSI: {umbral:.2f}")
    for nombre, medidas in escenarios.items():
        activaciones = medidas[list(FEATURES)].gt(umbral).any(axis=1)
        print(f"\n{nombre}: {activaciones.sum()}/{len(activaciones)} alertas")
        print(medidas.agg(["mean", "max"]).round(6).to_string())

    configuracion = {"ventana": ventana, "umbral": umbral, "referencia": referencia}
    args.salida.write_text(json.dumps(configuracion, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

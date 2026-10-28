"""Práctica de regresión de volumen a t+3; no clasifica demanda alta.

python3 comparar_proyecto.py
Tras guardar respuestas_llm_proyecto.txt (10 números en orden):
python3 comparar_proyecto.py --solo-llm
"""

import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
import runpy
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


CARPETA = Path(__file__).resolve().parent
# Tarifa API estándar de entrada, sin caché, consultada el 2026-10-28.
# https://developers.openai.com/api/docs/models/gpt-6.1-sol
PRECIO_ENTRADA = 2.00


def evaluar_llm(tabla, muestra):
    ruta = CARPETA / "respuestas_llm_proyecto.txt"
    if ruta.exists():
        predicciones = np.array([
            float(linea) for linea in ruta.read_text().splitlines() if linea.strip()
        ])
        if len(predicciones) != 10 or not np.isfinite(predicciones).all():
            raise ValueError("Se requieren 10 predicciones numéricas finitas, una por línea.")
        if (predicciones < 0).any():
            raise ValueError("El conteo de emergencias no puede ser negativo.")
        muestra["prediccion_llm"] = predicciones
        mae = mean_absolute_error(muestra["objetivo_real"], predicciones)
    else:
        mae = np.nan
        print("Pendiente: guardar 10 respuestas del LLM en", ruta, flush=True)

    texto = (CARPETA / "prompt_llm_proyecto.txt").read_text()
    costo = len(texto) / 4 * (10000 / 10) * PRECIO_ENTRADA / 1_000_000
    fila_llm = pd.DataFrame([{
        "modelo": "LLM GPT-6.1 Sol medium",
        "mae_entrenamiento": np.nan, "mae_prueba": np.nan,
        "mae_10_filas": mae, "segundos_entrenamiento": np.nan,
        "parametros_red": np.nan, "costo_entrada_10000_usd": costo,
    }])
    tabla = pd.concat([tabla.loc[~tabla["modelo"].str.startswith("LLM")], fila_llm], ignore_index=True)
    tabla.to_csv(CARPETA / "resultados_comparacion.csv", index=False, float_format="%.6f")
    muestra.to_csv(CARPETA / "evaluacion_10_filas.csv", index=False, float_format="%.6f")
    print(tabla.to_string(index=False), flush=True)
    print(f"Costo API de entrada estimado para 10 000 predicciones: ${costo:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solo-llm", action="store_true", help="Evaluar respuestas sin reentrenar.")
    opciones = parser.parse_args()
    if opciones.solo_llm:
        evaluar_llm(
            pd.read_csv(CARPETA / "resultados_comparacion.csv"),
            pd.read_csv(CARPETA / "evaluacion_10_filas.csv", dtype={"cod_canton": "string"}),
        )
        return

    ruta = CARPETA.parents[1] / "semana-09-sesion-02/plantilla-pipeline/pipeline_proyecto.py"
    with redirect_stdout(io.StringIO()):
        proyecto = runpy.run_path(str(ruta))
    X_ent, X_prueba = proyecto["X_ent"], proyecto["X_prueba"]
    y_ent, y_prueba = proyecto["y_ent"], proyecto["y_prueba"]
    datos = proyecto["datos"]

    # Selección fijada antes de conocer el error. Misma muestra para todos.
    muestra = X_prueba.sample(10, random_state=7)
    ejemplos = X_ent.sample(30, random_state=7)
    columnas = list(X_ent.columns)
    filas_ejemplo = datos.loc[ejemplos.index, ["fecha"] + columnas].copy()
    filas_ejemplo["emergencias_t3"] = y_ent.loc[ejemplos.index]
    filas_prompt = datos.loc[muestra.index, ["fecha"] + columnas]
    prompt = (
        "Predice el número de emergencias por cantón en t+3, al cierre del día t. "
        "Es regresión de volumen; no debes clasificar demanda alta ni calcular P90. "
        "fecha es el día t. Los lags y medias son históricos, cod_canton identifica "
        "el cantón y las banderas t3 describen el calendario de la fecha objetivo. "
        "Usa exclusivamente estos ejemplos. Responde con 10 números no negativos, "
        "uno por línea, en el orden de las filas sin objetivo, sin texto adicional.\n\n"
        "30 ejemplos de entrenamiento:\n" + filas_ejemplo.to_csv(index=False, float_format="%.4f")
        + "\n10 filas de prueba sin objetivo:\n" + filas_prompt.to_csv(index=False, float_format="%.4f")
    )
    (CARPETA / "prompt_llm_proyecto.txt").write_text(prompt)
    evaluacion = datos.loc[muestra.index, ["fecha", "cod_canton"]].copy()
    evaluacion.insert(0, "indice_original", muestra.index)
    evaluacion["objetivo_real"] = y_prueba.loc[muestra.index]
    print("Prompt listo:", CARPETA / "prompt_llm_proyecto.txt", flush=True)
    print(f"Train: {len(X_ent)} filas; prueba: {len(X_prueba)} filas.", flush=True)

    modelos = [
        ("bosque regularizado", RandomForestRegressor(
            n_estimators=100, max_depth=12, min_samples_leaf=20,
            random_state=42, n_jobs=2,
        )),
        # Escalar y ayuda a optimizar conteos con escalas muy diferentes.
        # Se ajusta solo con train y se revierte al predecir: el MAE conserva
        # las unidades originales de emergencias por cantón y día.
        ("red neuronal", TransformedTargetRegressor(
            regressor=MLPRegressor(
                hidden_layer_sizes=(64,), max_iter=300, batch_size=1024,
                early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=15, random_state=42,
            ), transformer=StandardScaler(),
        )),
    ]
    resultados = []
    for nombre, modelo in modelos:
        pipeline = Pipeline([("preparar", clone(proyecto["preparacion"])), ("modelo", modelo)])
        inicio = time.perf_counter()
        with threadpool_limits(limits=2):
            pipeline.fit(X_ent, y_ent)
        segundos = time.perf_counter() - inicio
        mae_ent = mean_absolute_error(y_ent, pipeline.predict(X_ent))
        mae_prueba = mean_absolute_error(y_prueba, pipeline.predict(X_prueba))
        predicciones = pipeline.predict(muestra)
        evaluacion["prediccion_" + nombre.replace(" ", "_")] = predicciones
        parametros = np.nan
        if nombre == "red neuronal":
            red = pipeline.named_steps["modelo"].regressor_
            parametros = sum(p.size for p in red.coefs_ + red.intercepts_)
            print(f"Red: {red.n_iter_} iteraciones, {parametros} parámetros.", flush=True)
        resultados.append({
            "modelo": nombre, "mae_entrenamiento": mae_ent, "mae_prueba": mae_prueba,
            "mae_10_filas": mean_absolute_error(y_prueba.loc[muestra.index], predicciones),
            "segundos_entrenamiento": segundos, "parametros_red": parametros,
            # No se midió una tarifa de infraestructura local: no se inventa.
            "costo_entrada_10000_usd": np.nan,
        })
        print(f"{nombre}: MAE prueba={mae_prueba:.4f}; entrenamiento={segundos:.2f} s", flush=True)
        # Conservar el resultado antes de empezar el siguiente entrenamiento.
        pd.DataFrame(resultados).to_csv(CARPETA / "resultados_comparacion.csv", index=False)
        evaluacion.to_csv(CARPETA / "evaluacion_10_filas.csv", index=False)

    evaluar_llm(pd.DataFrame(resultados), evaluacion)


if __name__ == "__main__":
    main()

"""Modelos del ejercicio 3 de Semana 10, Sesión 1.

Ejecutar desde cualquier carpeta: python3 ruta/a/modelos_proyecto.py
Reutiliza el pipeline y la división temporal del proyecto de Semana 9.
"""

from pathlib import Path
import runpy
import time

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline


def main():
    carpeta = Path(__file__).resolve().parent
    # El script original construye las features y separa fechas completas,
    # con tres días de separación para no mezclar etiquetas de train y test.
    ruta_pipeline = (
        carpeta.parents[1] / "semana-09-sesion-02"
        / "plantilla-pipeline" / "pipeline_proyecto.py"
    )
    proyecto = runpy.run_path(str(ruta_pipeline))
    X_ent, X_prueba = proyecto["X_ent"], proyecto["X_prueba"]
    y_ent, y_prueba = proyecto["y_ent"], proyecto["y_prueba"]

    modelos = [
        ("linea base mediana", DummyRegressor(strategy="median")),
        # Referencia adicional para medir el efecto de la penalización Ridge.
        ("lineal sin regularizar", LinearRegression()),
        ("ridge alpha=10", Ridge(alpha=10.0, solver="lsqr")),
        ("bosque regularizado", RandomForestRegressor(
            n_estimators=100, max_depth=12, min_samples_leaf=20,
            random_state=42, n_jobs=2,
        )),
    ]
    resultados = []
    for nombre, modelo in modelos:
        # clone elimina los parámetros aprendidos por el pipeline original.
        # Cada modelo vuelve a ajustar su preparación solo con entrenamiento.
        pipeline = Pipeline([
            ("preparar", clone(proyecto["preparacion"])),
            ("modelo", modelo),
        ])
        inicio = time.perf_counter()
        pipeline.fit(X_ent, y_ent)
        segundos = time.perf_counter() - inicio
        mae_ent = mean_absolute_error(y_ent, pipeline.predict(X_ent))
        mae_prueba = mean_absolute_error(y_prueba, pipeline.predict(X_prueba))
        resultados.append({
            "modelo": nombre,
            "mae_entrenamiento": mae_ent,
            "mae_prueba": mae_prueba,
            "brecha": mae_prueba - mae_ent,
            "segundos": segundos,
        })
        print(f"{nombre}: MAE train={mae_ent:.4f}, prueba={mae_prueba:.4f}", flush=True)

    tabla = pd.DataFrame(resultados)
    salida = carpeta / "resultados_proyecto.csv"
    tabla.to_csv(salida, index=False, float_format="%.6f")
    print("\n" + tabla.to_string(index=False, float_format=lambda valor: f"{valor:.4f}"))
    mejor = tabla.loc[tabla["mae_prueba"].idxmin()]
    error_base = tabla.iloc[0]["mae_prueba"]
    mejora = error_base - mejor["mae_prueba"]
    print(f"\nMejor modelo: {mejor['modelo']}")
    print(f"Reducción de MAE frente a la mediana: {mejora:.4f} ({100 * mejora / error_base:.2f}%)")
    print("Resultados guardados en:", salida)


if __name__ == "__main__":
    main()

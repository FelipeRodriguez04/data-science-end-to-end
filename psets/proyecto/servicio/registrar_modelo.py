"""python -m servicio.registrar_modelo --arboles 100"""
import argparse
import hashlib
import time
import platform
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from modelado.pipeline_proyecto import RUTA_DATOS, cargar_y_dividir, construir_preparacion

NOMBRE = "ecu911-volumen-emergencias-t3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arboles", type=int, default=100)
    args = parser.parse_args()
    if args.arboles < 1:
        parser.error("--arboles debe ser positivo")
    X_ent, X_prueba, y_ent, y_prueba = cargar_y_dividir()
    fechas = pd.read_csv(RUTA_DATOS, usecols=["fecha"], parse_dates=["fecha"])
    corte = fechas.loc[X_prueba.index, "fecha"].min().date().isoformat()
    huella = hashlib.sha256(RUTA_DATOS.read_bytes()).hexdigest()
    modelo = Pipeline([("preparar", construir_preparacion()), ("modelo", RandomForestRegressor(
        n_estimators=args.arboles, max_depth=12, min_samples_leaf=20, random_state=42, n_jobs=2))])
    mlflow.set_experiment(NOMBRE + "-registro")
    with mlflow.start_run(run_name=f"bosque-{args.arboles}") as corrida:
        mlflow.log_params({"arboles": args.arboles, "max_depth": 12, "min_samples_leaf": 20,
            "random_state": 42, "huella_sha256": huella, "fecha_corte": corte, "separacion_dias": 3,
            "filas_entrenamiento": len(X_ent), "filas_prueba": len(X_prueba),
            "sklearn": sklearn.__version__, "python": platform.python_version()})
        inicio = time.perf_counter()
        with threadpool_limits(limits=2):
            modelo.fit(X_ent, y_ent)
        mlflow.log_metrics({"mae": mean_absolute_error(y_prueba, modelo.predict(X_prueba)),
            "mae_entrenamiento": mean_absolute_error(y_ent, modelo.predict(X_ent)),
            "segundos": time.perf_counter() - inicio})
        mlflow.log_artifact(str(Path(__file__).resolve().parents[1] / "modelado/pipeline_proyecto.py"), "codigo")
        registro = mlflow.sklearn.log_model(modelo, "modelo", registered_model_name=NOMBRE)
        print(f"Modelo {NOMBRE}; versión {registro.registered_model_version}; run_id {corrida.info.run_id}", flush=True)


if __name__ == "__main__":
    main()

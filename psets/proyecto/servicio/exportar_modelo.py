"""Desde psets/proyecto: python servicio/exportar_modelo.py [--run-id ID]."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import sklearn

RUN_ID = "0c40681fa7d54d56a75a95cd8ee29f5e"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    run = mlflow.MlflowClient().get_run(args.run_id)
    if run.info.status != "FINISHED":
        raise ValueError("La corrida debe estar finalizada")
    if run.data.params.get("sklearn") != sklearn.__version__:
        raise ValueError("Usa la misma versión de scikit-learn de la corrida")
    modelo = mlflow.sklearn.load_model(f"runs:/{args.run_id}/modelo")
    carpeta = Path(__file__).resolve().parent
    ruta = carpeta / "modelo.joblib"
    joblib.dump(modelo, ruta)
    metadata = {"run_id": args.run_id, "sklearn": sklearn.__version__,
                "sha256": hashlib.sha256(ruta.read_bytes()).hexdigest()}
    (carpeta / "modelo_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print("Modelo exportado:", ruta)


if __name__ == "__main__":
    main()

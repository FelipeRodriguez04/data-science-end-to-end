"""Desde psets/proyecto: python -m servicio.exportar_modelo (exporta el alias produccion)."""
import hashlib
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import sklearn


def main():
    from servicio.registrar_modelo import NOMBRE
    cliente = mlflow.MlflowClient()
    version = cliente.get_model_version_by_alias(NOMBRE, "produccion")
    run = cliente.get_run(version.run_id)
    if run.info.status != "FINISHED":
        raise ValueError("La corrida debe estar finalizada")
    if run.data.params.get("sklearn") != sklearn.__version__:
        raise ValueError("Usa la misma versión de scikit-learn de la corrida")
    modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE}/{version.version}")
    carpeta = Path(__file__).resolve().parent
    ruta = carpeta / "modelo.joblib"
    joblib.dump(modelo, ruta)
    metadata = {"nombre_modelo": NOMBRE, "version_modelo": version.version, "run_id": version.run_id, "sklearn": sklearn.__version__,
                "sha256": hashlib.sha256(ruta.read_bytes()).hexdigest()}
    (carpeta / "modelo_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (carpeta / "modelo_version.txt").write_text(version.version + "\n")
    print(json.dumps(metadata, indent=2))
    print("Modelo exportado:", ruta)


if __name__ == "__main__":
    main()

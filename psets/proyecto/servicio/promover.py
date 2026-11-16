"""python -m servicio.promover VERSION --responsable NOMBRE --motivo TEXTO [--rollback]"""
import argparse
from datetime import datetime, timezone

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from servicio.registrar_modelo import NOMBRE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("--responsable", required=True)
    parser.add_argument("--motivo", required=True)
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    cliente = MlflowClient()
    candidata = cliente.get_model_version(NOMBRE, args.version)
    corrida = cliente.get_run(candidata.run_id)
    try:
        actual = cliente.get_model_version_by_alias(NOMBRE, "produccion")
    except MlflowException as error:
        if not (error.error_code == "RESOURCE_DOES_NOT_EXIST" or (error.error_code == "INVALID_PARAMETER_VALUE" and "alias produccion not found" in str(error))):
            raise
        actual = None
    aprobada = True
    if actual is not None and not args.rollback:
        anterior = cliente.get_run(actual.run_id)
        claves = ["huella_sha256", "fecha_corte", "separacion_dias", "filas_prueba"]
        comparables = all(corrida.data.params[k] == anterior.data.params[k] for k in claves)
        aprobada = comparables and corrida.data.metrics["mae"] < anterior.data.metrics["mae"]
    etiquetas = {"validacion": "aprobada" if aprobada else "rechazada",
        "comparada_con": actual.version if actual else "ninguna", "responsable": args.responsable,
        "motivo": args.motivo, "fecha_decision_utc": datetime.now(timezone.utc).isoformat(),
        "tipo_decision": "rollback" if args.rollback else "promocion"}
    for clave, valor in etiquetas.items():
        cliente.set_model_version_tag(NOMBRE, args.version, clave, valor)
    if aprobada:
        cliente.set_registered_model_alias(NOMBRE, "produccion", args.version)
    print({"version": args.version, "promovida": aprobada, "mae": corrida.data.metrics["mae"], **etiquetas})


if __name__ == "__main__":
    main()

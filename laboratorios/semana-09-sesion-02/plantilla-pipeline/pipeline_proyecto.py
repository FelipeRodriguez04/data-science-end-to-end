"""Compatibilidad de los laboratorios con el pipeline canónico del proyecto."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_ruta = Path(__file__).resolve().parents[3] / "psets/proyecto/modelado/pipeline_proyecto.py"
_spec = spec_from_file_location("_pipeline_ecu911", _ruta)
_pipeline = module_from_spec(_spec)
_spec.loader.exec_module(_pipeline)

RUTA_DATOS = _pipeline.RUTA_DATOS
OBJETIVO = _pipeline.OBJETIVO
NUMERICAS = _pipeline.NUMERICAS
CATEGORICAS = _pipeline.CATEGORICAS
YA_LISTAS = _pipeline.YA_LISTAS
agregar_features = _pipeline.agregar_features
cargar_y_dividir = _pipeline.cargar_y_dividir
construir_preparacion = _pipeline.construir_preparacion

# Mantiene las variables que consumen los ejercicios anteriores con runpy.
import pandas as pd

# Conserva compatibilidad con los ejercicios que usan runpy.run_path.
if __name__ in ("__main__", "<run_path>"):
    datos = agregar_features(pd.read_csv(
        RUTA_DATOS,
        dtype={"cod_canton": "string", "cod_provincia": "string"},
        parse_dates=["fecha"],
    ))
    X_ent, X_prueba, y_ent, y_prueba = cargar_y_dividir()
    preparacion = construir_preparacion()
    X_ent_listo = preparacion.fit_transform(X_ent)
    X_prueba_listo = preparacion.transform(X_prueba)
    print("Entrenamiento:", X_ent_listo.shape, "Prueba:", X_prueba_listo.shape)
    print(list(preparacion.get_feature_names_out()))
    print("\nMedianas aprendidas solo con entrenamiento:")
    print(pd.Series(
        preparacion.named_transformers_["numericas"].named_steps["imputar"].statistics_,
        index=NUMERICAS,
    ))
    # Los faltantes iniciales de las ventanas se imputan con medianas de train;
    # los objetivos ausentes nunca se imputan. En prueba se usa solo transform.

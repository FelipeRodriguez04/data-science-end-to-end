"""Entrada del proyecto al pipeline validado en los laboratorios de ECU911.

Importarlo no lee datos ni entrena. Se reutiliza una única implementación.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

RUTA_IMPLEMENTACION = (
    Path(__file__).resolve().parents[3] / "laboratorios" / "semana-09-sesion-02"
    / "plantilla-pipeline" / "pipeline_proyecto.py"
)
_spec = spec_from_file_location("_pipeline_ecu911", RUTA_IMPLEMENTACION)
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

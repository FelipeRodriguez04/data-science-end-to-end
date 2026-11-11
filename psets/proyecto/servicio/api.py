"""API de volumen exploratorio de emergencias por cantón a t+3."""
import hashlib
import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from modelado.pipeline_proyecto import agregar_features, NUMERICAS, CATEGORICAS, YA_LISTAS

RUTA_MODELO = Path(os.environ.get("RUTA_MODELO", str(Path(__file__).with_name("modelo.joblib"))))
modelo = None
version_modelo = None
run_id = None
if RUTA_MODELO.exists():
    version_modelo = hashlib.sha256(RUTA_MODELO.read_bytes()).hexdigest()
    metadata_path = RUTA_MODELO.with_name("modelo_metadata.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["sha256"] != version_modelo:
            raise ValueError("La huella del modelo no coincide con sus metadatos")
        run_id = metadata["run_id"]
    modelo = joblib.load(RUTA_MODELO)

app = FastAPI(title="ECU911: volumen de emergencias a t+3")


class CantonDia(BaseModel):
    """Históricos disponibles al cierre de fecha y calendario objetivo."""
    model_config = ConfigDict(allow_inf_nan=False)
    fecha: date
    cod_canton: str = Field(pattern=r"^\d{4}$")
    n_emergencias: float = Field(ge=0)
    lag_1d: float | None = Field(default=None, ge=0)
    lag_7d: float | None = Field(default=None, ge=0)
    lag_14d: float | None = Field(default=None, ge=0)
    lag_28d: float | None = Field(default=None, ge=0)
    media_prev_7d: float | None = Field(default=None, ge=0)
    media_prev_28d: float | None = Field(default=None, ge=0)
    es_feriado_t3: Literal[0, 1]


@app.get("/salud")
def salud():
    return {"estado": "ok", "modelo_cargado": modelo is not None,
            "version_modelo": version_modelo, "run_id": run_id}


@app.post("/predecir")
def predecir(entrada: CantonDia):
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible: ejecuta exportar_modelo.py")
    datos = pd.DataFrame([entrada.model_dump()])
    datos["fecha"] = pd.to_datetime(datos["fecha"])
    datos = agregar_features(datos)
    prediccion = float(modelo.predict(datos[NUMERICAS + CATEGORICAS + YA_LISTAS])[0])
    return {"cod_canton": entrada.cod_canton, "fecha_corte": entrada.fecha.isoformat(),
            "fecha_objetivo": (entrada.fecha + timedelta(days=3)).isoformat(),
            "horizonte_dias": 3, "emergencias_estimadas": round(prediccion, 4),
            "version_modelo": version_modelo, "run_id": run_id}

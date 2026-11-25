"""API de volumen exploratorio de emergencias por cantón a t+3."""
import hashlib
import json
import os
from collections import deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Literal
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from modelado.pipeline_proyecto import agregar_features, NUMERICAS, CATEGORICAS, YA_LISTAS
from servicio.deriva import FEATURES, MonitorDeDeriva

RUTA_MODELO = Path(os.environ.get("RUTA_MODELO", str(Path(__file__).with_name("modelo.joblib"))))
modelo = None
version_modelo = None
run_id = None
huella_modelo = None
if RUTA_MODELO.exists():
    huella_modelo = hashlib.sha256(RUTA_MODELO.read_bytes()).hexdigest()
    metadata_path = RUTA_MODELO.with_name("modelo_metadata.json")
    metadata = json.loads(metadata_path.read_text())
    version_modelo = RUTA_MODELO.with_name("modelo_version.txt").read_text().strip()
    if metadata["sha256"] != huella_modelo or metadata["version_modelo"] != version_modelo:
        raise ValueError("El modelo, la versión y los metadatos no coinciden; vuelve a exportar")
    run_id = metadata["run_id"]
    modelo = joblib.load(RUTA_MODELO)

app = FastAPI(title="ECU911: volumen de emergencias a t+3")

RUTA_REFERENCIA = Path(os.environ.get(
    "RUTA_REFERENCIA_DERIVA", Path(__file__).with_name("referencia_deriva.json")
))
app.state.monitor_deriva = (
    MonitorDeDeriva(json.loads(RUTA_REFERENCIA.read_text(encoding="utf-8")))
    if RUTA_REFERENCIA.is_file() else None
)

RUTA_PREDICCIONES = Path(os.environ.get(
    "RUTA_PREDICCIONES", Path(__file__).resolve().parents[1] / "monitoreo/predicciones.jsonl"
))
VENTANA_LATENCIA = 10_000
app.state.latencias = deque(maxlen=VENTANA_LATENCIA)
app.state.conteos = {"peticiones": 0, "errores_validacion": 0, "errores_servidor": 0}
candado_metricas = Lock()
candado_registro = Lock()


@app.middleware("http")
async def medir_prediccion(request: Request, call_next):
    if request.method != "POST" or request.url.path != "/predecir":
        return await call_next(request)

    inicio = perf_counter()
    estado = 500
    try:
        respuesta = await call_next(request)
        estado = respuesta.status_code
        return respuesta
    finally:
        latencia_ms = (perf_counter() - inicio) * 1_000
        with candado_metricas:
            app.state.conteos["peticiones"] += 1
            app.state.conteos["errores_validacion"] += int(estado == 422)
            app.state.conteos["errores_servidor"] += int(500 <= estado < 600)
            app.state.latencias.append(latencia_ms)


@app.get("/metricas")
def metricas():
    with candado_metricas:
        conteos = app.state.conteos.copy()
        latencias = pd.Series(list(app.state.latencias), dtype=float)
    return {
        **conteos,
        "muestras_latencia": len(latencias),
        "latencia_p50_ms": round(float(latencias.quantile(0.50)), 3) if len(latencias) else None,
        "latencia_p95_ms": round(float(latencias.quantile(0.95)), 3) if len(latencias) else None,
    }


@app.get("/deriva")
def deriva():
    monitor = app.state.monitor_deriva
    if monitor is None:
        raise HTTPException(status_code=503, detail="Referencia no disponible: ejecuta monitoreo.simular_deriva")
    return monitor.estado()


def registrar_prediccion(entrada, salida):
    evento = {
        "id_prediccion": salida["id_prediccion"],
        "momento_utc": datetime.now(timezone.utc).isoformat(),
        "version_modelo": salida["version_modelo"],
        "entrada": entrada.model_dump(mode="json"),
        "salida": salida,
    }
    with candado_registro:
        RUTA_PREDICCIONES.parent.mkdir(parents=True, exist_ok=True)
        with RUTA_PREDICCIONES.open("a", encoding="utf-8") as archivo:
            archivo.write(json.dumps(evento, ensure_ascii=False, allow_nan=False) + "\n")


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
            "version_modelo": version_modelo, "run_id": run_id, "huella_modelo": huella_modelo}


@app.post("/predecir")
def predecir(entrada: CantonDia):
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible: ejecuta exportar_modelo.py")
    datos = pd.DataFrame([entrada.model_dump()])
    datos["fecha"] = pd.to_datetime(datos["fecha"])
    datos = agregar_features(datos)
    prediccion = float(modelo.predict(datos[NUMERICAS + CATEGORICAS + YA_LISTAS])[0])
    salida = {"id_prediccion": uuid4().hex, "cod_canton": entrada.cod_canton, "fecha_corte": entrada.fecha.isoformat(),
            "fecha_objetivo": (entrada.fecha + timedelta(days=3)).isoformat(),
            "horizonte_dias": 3, "emergencias_estimadas": round(prediccion, 4),
            "version_modelo": version_modelo, "run_id": run_id, "huella_modelo": huella_modelo}
    registrar_prediccion(entrada, salida)
    if app.state.monitor_deriva is not None:
        app.state.monitor_deriva.registrar(entrada.model_dump(include=set(FEATURES)))
    return salida

"""Servicio de prediccion de la tarifa, instrumentado.

Ademas de responder la tarifa, mide cuanto tarda cada prediccion y con
que codigo responde (las metricas del sistema, en /metricas), y guarda
cada prediccion en predicciones.jsonl, para medir despues cuanto se
equivoco el modelo cuando llegue la tarifa cobrada (desempeno.py).

    uvicorn servicio:app --port 8000
"""

import json
import os
import time
import uuid
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel

from features_taxis import agregar_features
from preparacion_taxis import CATEGORICAS, NUMERICAS, YA_LISTAS

RUTA_MODELO = "modelo_tarifa.joblib"
RUTA_VERSION = "modelo_version.txt"
RUTA_REGISTRO = os.environ.get("RUTA_REGISTRO", "predicciones.jsonl")

modelo = None
version_modelo = None
if os.path.exists(RUTA_MODELO):
    modelo = joblib.load(RUTA_MODELO)
    with open(RUTA_VERSION) as archivo:
        version_modelo = archivo.read().strip()


class Metricas:
    """Las metricas del sistema, en memoria: una latencia y un codigo por
    cada peticion a /predecir."""

    def __init__(self):
        self.latencias_ms = []
        self.codigos = []

    def registrar(self, latencia_ms, codigo):
        self.latencias_ms.append(latencia_ms)
        self.codigos.append(codigo)

    def resumen(self):

        p95=p99=None
        error_validacion=error_servidor=0

        if self.latencias_ms:
            p95 = round(np.percentile(self.latencias_ms, 95), 2)
            p99 = round(np.percentile(self.latencias_ms, 99), 2)

        for codigo in self.codigos:
            if codigo==422:
                error_validacion += 1
            elif codigo==500:
                error_servidor += 1

        return {
            'volumen_peticiones': len(self.codigos),
            'error_validacion_datos': error_validacion,
            'error_servidor': error_servidor,
            'latencia_p95_ms': p95,
            'latencia_p99_ms': p99,
        }


metricas = Metricas()
app = FastAPI(title="Tarifa estimada de taxis")


@app.middleware("http")
async def medir(request: Request, call_next):
    """Corre alrededor de cada peticion: lo que esta antes de call_next
    pasa antes de la ruta, y lo que esta despues, cuando ya hay respuesta."""
    if request.url.path != "/predecir":
        return await call_next(request)

    inicio=time.perf_counter()

    resultado = await call_next(request)
    tiempo_respuesta=(time.perf_counter() - inicio) *1000

    metricas.registrar(tiempo_respuesta, resultado.status_code)

    return resultado


class Viaje(BaseModel):
    """Lo que la app sabe de un viaje antes de que empiece."""
    fecha_hora_inicio: datetime
    zona_origen: str
    zona_destino: str
    distancia_km: float | None = None
    pasajeros: float | None = None
    metodo_pago: str | None = None


@app.get("/salud")
def salud():
    return {"estado": "ok", "modelo_cargado": modelo is not None, "version_modelo": version_modelo}


@app.get("/metricas")
def ver_metricas():
    return metricas.resumen()


@app.post("/predecir")
def predecir(viaje: Viaje):
    datos = agregar_features(pd.DataFrame([viaje.model_dump()]))
    tarifa = round(float(modelo.predict(datos[NUMERICAS + YA_LISTAS + CATEGORICAS])[0]), 2)
    id_prediccion = uuid.uuid4().hex

    registro_predicciones = {
        "id_prediccion": id_prediccion,
        "prediccion_created_at": datetime.now().isoformat(),
        "version_modelo": version_modelo,
        "request_data": viaje.model_dump_json(),
        "tarifa_estimada": tarifa,
    }

    with open(RUTA_REGISTRO, "a") as archivo:
        archivo.write(json.dumps(registro_predicciones) + "\n")

    return {"id_prediccion": id_prediccion, "tarifa_estimada": tarifa, "version_modelo": version_modelo}

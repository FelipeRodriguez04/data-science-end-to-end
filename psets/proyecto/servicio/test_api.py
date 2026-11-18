"""Contrato HTTP con modelo sintético; no necesita MLflow ni datos reales."""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from servicio import api
from modelado.pipeline_proyecto import agregar_features, construir_preparacion, NUMERICAS, CATEGORICAS, YA_LISTAS

ENTRADA = {
    "fecha": "2026-11-05", "cod_canton": "0101", "n_emergencias": 12,
    "lag_1d": 10, "lag_7d": 11, "lag_14d": 9, "lag_28d": 8,
    "media_prev_7d": 10, "media_prev_28d": 5, "es_feriado_t3": 0,
}


@pytest.fixture
def cliente(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "RUTA_PREDICCIONES", tmp_path / "predicciones.jsonl")
    monkeypatch.setattr(api.app.state, "latencias", api.deque(maxlen=api.VENTANA_LATENCIA))
    monkeypatch.setattr(api.app.state, "conteos", dict.fromkeys(api.app.state.conteos, 0))
    datos = pd.DataFrame([ENTRADA, ENTRADA])
    datos["fecha"] = pd.to_datetime(datos["fecha"])
    datos = agregar_features(datos)
    modelo = Pipeline([("preparar", construir_preparacion()), ("modelo", DummyRegressor(strategy="constant", constant=20.0))])
    modelo.fit(datos[NUMERICAS + CATEGORICAS + YA_LISTAS], [20., 20.])
    monkeypatch.setattr(api, "modelo", modelo)
    monkeypatch.setattr(api, "version_modelo", "7")
    monkeypatch.setattr(api, "run_id", "corrida-prueba")
    monkeypatch.setattr(api, "huella_modelo", "sha-prueba")
    with TestClient(api.app) as cliente:
        yield cliente


def test_salud(cliente):
    respuesta = cliente.get("/salud")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"estado": "ok", "modelo_cargado": True, "version_modelo": "7", "run_id": "corrida-prueba", "huella_modelo": "sha-prueba"}


def test_prediccion_valida(cliente):
    respuesta = cliente.post("/predecir", json=ENTRADA)
    assert respuesta.status_code == 200
    salida = respuesta.json()
    assert salida.pop("id_prediccion")
    assert salida == {"cod_canton": "0101", "fecha_corte": "2026-11-05", "fecha_objetivo": "2026-11-08", "horizonte_dias": 3, "emergencias_estimadas": 20.0, "version_modelo": "7", "run_id": "corrida-prueba", "huella_modelo": "sha-prueba"}


@pytest.mark.parametrize("cambios", [{"fecha": "incorrecta"}, {"cod_canton": "101"}, {"n_emergencias": -1}, {"es_feriado_t3": 2}])
def test_entrada_invalida_no_llega_al_modelo(cliente, monkeypatch, cambios):
    def inesperado(*args, **kwargs):
        pytest.fail("La entrada inválida no debe llegar a predict")
    monkeypatch.setattr(api.modelo, "predict", inesperado)
    respuesta = cliente.post("/predecir", json={**ENTRADA, **cambios})
    assert respuesta.status_code == 422


def test_servicio_calcula_features_de_entrenamiento(cliente, monkeypatch):
    def predecir(datos):
        assert datos.loc[0, "demanda_relativa_7d_28d"] == 2.0
        assert datos.loc[0, "es_fin_de_semana_t3"] == 1  # Domingo 8.
        assert "fecha" not in datos.columns
        return np.array([20.])
    monkeypatch.setattr(api.modelo, "predict", predecir)
    assert cliente.post("/predecir", json=ENTRADA).status_code == 200


def test_sin_modelo_devuelve_503(cliente, monkeypatch):
    monkeypatch.setattr(api, "modelo", None)
    assert cliente.get("/salud").json()["modelo_cargado"] is False
    assert cliente.post("/predecir", json=ENTRADA).status_code == 503


def test_prediccion_incluye_version_desplegada(cliente):
    respuesta = cliente.post("/predecir", json=ENTRADA)
    assert respuesta.status_code == 200
    assert respuesta.json()["version_modelo"] == "7"
    assert cliente.get("/salud").json()["version_modelo"] == "7"



def test_prediccion_queda_registrada(cliente):
    primera = cliente.post("/predecir", json=ENTRADA).json()
    segunda = cliente.post("/predecir", json=ENTRADA).json()
    eventos = [json.loads(linea) for linea in api.RUTA_PREDICCIONES.read_text().splitlines()]
    assert len(eventos) == 2
    assert primera["id_prediccion"] != segunda["id_prediccion"]
    for evento, salida in zip(eventos, [primera, segunda]):
        assert evento["id_prediccion"] == salida["id_prediccion"]
        assert evento["version_modelo"] == "7"
        assert evento["momento_utc"].endswith("+00:00")
        assert evento["entrada"] == ENTRADA
        assert evento["salida"] == salida


def test_metricas_separan_validacion_y_servidor(cliente, monkeypatch):
    assert cliente.get("/metricas").json()["latencia_p95_ms"] is None
    assert cliente.post("/predecir", json=ENTRADA).status_code == 200
    assert cliente.post("/predecir", json={}).status_code == 422
    monkeypatch.setattr(api, "modelo", None)
    assert cliente.post("/predecir", json=ENTRADA).status_code == 503
    respuesta = cliente.get("/metricas").json()
    assert respuesta["peticiones"] == respuesta["muestras_latencia"] == 3
    assert respuesta["errores_validacion"] == respuesta["errores_servidor"] == 1
    assert 0 <= respuesta["latencia_p50_ms"] <= respuesta["latencia_p95_ms"]
    assert len(api.RUTA_PREDICCIONES.read_text().splitlines()) == 1

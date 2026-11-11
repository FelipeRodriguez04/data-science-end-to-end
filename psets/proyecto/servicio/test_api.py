"""Contrato HTTP con modelo sintético; no necesita MLflow ni datos reales."""
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
def cliente(monkeypatch):
    datos = pd.DataFrame([ENTRADA, ENTRADA])
    datos["fecha"] = pd.to_datetime(datos["fecha"])
    datos = agregar_features(datos)
    modelo = Pipeline([("preparar", construir_preparacion()), ("modelo", DummyRegressor(strategy="constant", constant=20.0))])
    modelo.fit(datos[NUMERICAS + CATEGORICAS + YA_LISTAS], [20., 20.])
    monkeypatch.setattr(api, "modelo", modelo)
    monkeypatch.setattr(api, "version_modelo", "modelo-prueba")
    monkeypatch.setattr(api, "run_id", "corrida-prueba")
    with TestClient(api.app) as cliente:
        yield cliente


def test_salud(cliente):
    respuesta = cliente.get("/salud")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"estado": "ok", "modelo_cargado": True, "version_modelo": "modelo-prueba", "run_id": "corrida-prueba"}


def test_prediccion_valida(cliente):
    respuesta = cliente.post("/predecir", json=ENTRADA)
    assert respuesta.status_code == 200
    assert respuesta.json() == {"cod_canton": "0101", "fecha_corte": "2026-11-05", "fecha_objetivo": "2026-11-08", "horizonte_dias": 3, "emergencias_estimadas": 20.0, "version_modelo": "modelo-prueba", "run_id": "corrida-prueba"}


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

"""Contratos del pipeline con datos sintéticos; no necesita data/ ni MLflow."""
import numpy as np
import pandas as pd
from modelado.pipeline_proyecto import (
    OBJETIVO, NUMERICAS, CATEGORICAS, YA_LISTAS,
    agregar_features, cargar_y_dividir, construir_preparacion,
)


def ejemplo():
    fechas = pd.date_range("2026-01-01", periods=30)
    n = len(fechas)
    return pd.DataFrame({
        "fecha": fechas, "cod_canton": ["0101", "0102"] * 15,
        "cod_provincia": "01", "n_emergencias": np.arange(n) + 10,
        "lag_1d": 10., "lag_7d": 9., "lag_14d": 8., "lag_28d": 7.,
        "media_prev_7d": 6., "media_prev_28d": 3.,
        "es_feriado_t3": 0, OBJETIVO: np.arange(n) + 20.,
    })


def test_objetivo_excluido_de_entradas():
    preparacion = construir_preparacion()
    columnas = [c for _, _, grupo in preparacion.transformers for c in grupo]
    assert OBJETIVO not in columnas
    assert not any(c.startswith("obj_") for c in columnas)
    assert columnas == NUMERICAS + CATEGORICAS + YA_LISTAS


def test_features_con_valores_conocidos_sin_modificar_original():
    datos = ejemplo()
    original = datos.copy(deep=True)
    salida = agregar_features(datos)
    assert salida.loc[0, "demanda_relativa_7d_28d"] == 2.0
    # Jueves 1 + 3 días = domingo 4; viernes 2 + 3 = lunes 5.
    assert salida.loc[0, "es_fin_de_semana_t3"] == 1
    assert salida.loc[1, "es_fin_de_semana_t3"] == 0
    pd.testing.assert_frame_equal(datos, original)


def test_media_base_cero_no_produce_infinito():
    datos = ejemplo()
    datos.loc[0, "media_prev_28d"] = 0
    salida = agregar_features(datos)
    assert pd.isna(salida.loc[0, "demanda_relativa_7d_28d"])
    assert not np.isinf(salida["demanda_relativa_7d_28d"]).any()


def test_preparacion_acepta_canton_desconocido_y_faltantes():
    datos = agregar_features(ejemplo())
    preparacion = construir_preparacion()
    entrenamiento = preparacion.fit_transform(datos)
    nuevo = datos.iloc[[0]].copy()
    nuevo["cod_canton"] = "9999"
    nuevo["lag_1d"] = np.nan
    salida = preparacion.transform(nuevo)
    assert salida.shape == (1, entrenamiento.shape[1])
    assert np.isfinite(salida.toarray()).all()
    assert preparacion.named_transformers_["numericas"].named_steps["imputar"].statistics_[1] == 10.0


def test_division_temporal_purga_etiquetas_y_preserva_fechas(tmp_path):
    datos = pd.concat([ejemplo(), ejemplo().assign(cod_canton="0201")], ignore_index=True)
    datos.loc[datos.fecha == datos.fecha.max(), OBJETIVO] = np.nan
    ruta = tmp_path / "sintetico.csv"
    datos.to_csv(ruta, index=False)
    X_ent, X_prueba, y_ent, y_prueba = cargar_y_dividir(ruta)
    assert len(X_ent) > 0 and len(X_prueba) > 0
    assert datos.loc[X_ent.index, "fecha"].max() + pd.Timedelta(days=3) < datos.loc[X_prueba.index, "fecha"].min()
    assert set(datos.loc[X_ent.index, "fecha"]).isdisjoint(datos.loc[X_prueba.index, "fecha"])
    assert X_ent.index.equals(y_ent.index) and X_prueba.index.equals(y_prueba.index)
    assert y_ent.notna().all() and y_prueba.notna().all()
    assert OBJETIVO not in X_ent.columns
    assert X_ent["cod_canton"].str.startswith("0").all()

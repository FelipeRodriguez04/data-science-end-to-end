"""PSI de los conteos recientes de ECU911."""

from collections import deque
from threading import Lock

import numpy as np

FEATURES = ("n_emergencias", "media_prev_7d")
MINIMO = 0.0001


def proporciones(valores, cortes):
    """Histograma con extremos abiertos y una casilla para faltantes."""
    valores = np.asarray(valores, dtype=float)
    presentes = ~np.isnan(valores)
    conteos, _ = np.histogram(valores[presentes], bins=[-np.inf, *cortes, np.inf])
    return np.append(conteos, np.count_nonzero(~presentes)) / len(valores)


def medir_psi(columnas, referencia):
    resultado = {}
    for feature, distribucion in referencia.items():
        esperada = np.clip(distribucion["proporciones"], MINIMO, None)
        actual = np.clip(proporciones(columnas[feature], distribucion["cortes"]), MINIMO, None)
        resultado[feature] = float(np.sum((actual - esperada) * np.log(actual / esperada)))
    return resultado


class MonitorDeDeriva:
    def __init__(self, configuracion):
        self.referencia = configuracion["referencia"]
        self.ventana = configuracion["ventana"]
        self.umbral = configuracion["umbral"]
        self.filas = deque(maxlen=self.ventana)
        self.candado = Lock()

    def registrar(self, entrada):
        fila = tuple(entrada[feature] for feature in FEATURES)
        with self.candado:
            self.filas.append(fila)

    def estado(self):
        with self.candado:
            filas = list(self.filas)
        columnas = dict(zip(FEATURES, zip(*filas)))
        valores = medir_psi(columnas, self.referencia) if filas else {}
        return {
            "predicciones_en_ventana": len(filas),
            "ventana": self.ventana,
            "umbral": self.umbral,
            "psi": {feature: round(valor, 6) for feature, valor in valores.items()},
            "alerta": len(filas) == self.ventana and any(valor > self.umbral for valor in valores.values()),
        }

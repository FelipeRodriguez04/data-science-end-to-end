"""El monitor de deriva de datos del servicio.

Guarda los ultimos `ventana` viajes que recibio el servicio y compara la
distribucion de distancia_km y de zona_origen con la de entrenamiento
(referencia_deriva.json), con el PSI del notebook del Ejercicio 1.

Despues de resolver, corre python comprobar.py hasta que los cinco pasos
digan OK, y despues python simular_semanas.py.
"""

from collections import Counter
from threading import Lock

import numpy as np

MINIMO = 0.0001


def proporciones_numericas(valores, cortes):
    """Proporcion de los valores en cada casilla. `cortes` son los limites
    interiores: con 9 cortes hay 10 casillas, la primera hasta el primer
    corte y la ultima desde el ultimo, abiertas. Los None no cuentan."""
    presentes = [valor for valor in valores if valor is not None and not np.isnan(valor)]
    conteo, _ = np.histogram(presentes, bins=[-np.inf, *cortes, np.inf])
    return conteo / len(presentes) if presentes else conteo.astype(float)


def proporciones_categoricas(valores, categorias):
    """Proporcion de los valores en cada categoria, en el orden de
    `categorias`. Un valor que no esta en `categorias` no cuenta en ninguna
    casilla, pero si en el total."""
    conteo = Counter(valores)
    frecuencias = np.array([conteo[categoria] for categoria in categorias], dtype=float)
    return frecuencias / len(valores) if len(valores) else frecuencias


def psi(referencia, actual):
    """El PSI entre dos listas de proporciones, con MINIMO en las casillas
    vacias, como en el notebook."""
    referencia = np.clip(referencia, MINIMO, None)
    actual = np.clip(actual, MINIMO, None)
    return float(np.sum((actual - referencia) * np.log(actual / referencia)))


class MonitorDeDeriva:
    def __init__(self, referencia, ventana=300, umbral=0.15):
        self.referencia = referencia
        self.ventana = ventana
        self.umbral = umbral
        self.distancias = []
        self.zonas = []
        self.candado = Lock()

    def registrar(self, viaje):
        """Agrega un viaje (un diccionario con distancia_km y zona_origen) y
        deja solo los ultimos `ventana`."""
        distancia, zona = viaje["distancia_km"], viaje["zona_origen"]
        with self.candado:
            self.distancias.append(distancia)
            self.zonas.append(zona)
            del self.distancias[:-self.ventana]
            del self.zonas[:-self.ventana]

    def estado(self):
        """Un diccionario con "viajes_en_ventana", "psi" (un diccionario con
        el PSI de distancia_km y de zona_origen, redondeado a 4 decimales) y
        "alerta": True solo si la ventana esta llena y alguno de los dos PSI
        es mayor que el umbral. Con la ventana vacia, "psi" es un diccionario
        vacio."""
        with self.candado:
            distancias = self.distancias.copy()
            zonas = self.zonas.copy()

        valores = {}
        if zonas:
            distancia = self.referencia["distancia_km"]
            zona = self.referencia["zona_origen"]
            valores["distancia_km"] = psi(
                distancia["proporciones"], proporciones_numericas(distancias, distancia["cortes"])
            )
            valores["zona_origen"] = psi(
                zona["proporciones"], proporciones_categoricas(zonas, zona["categorias"])
            )
        return {
            "viajes_en_ventana": len(zonas),
            "psi": {feature: round(valor, 4) for feature, valor in valores.items()},
            "alerta": len(zonas) == self.ventana and any(valor > self.umbral for valor in valores.values()),
        }

# Semana 10, Sesión 1: Entrenamiento y comparación de modelos I: línea base y ensamblajes

> Objetivo de la sesión: entrenar un modelo de línea base y un modelo de ensamblaje sobre los datos preparados de tu proyecto, y aplicar regularización cuando el modelo se sobreajusta, sin derivar la matemática del descenso de gradiente.

**Lectura previa de esta sesión:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

**Referencia de esta sesión:** [Ensemble methods (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/ensemble.html).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn, que ya usaste la sesión pasada. Ten a mano el pipeline de tu proyecto de la Semana 9, Sesión 2.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es un ensamblaje, y en qué se diferencian bagging y boosting en la forma de combinar sus modelos?**

_(tu respuesta)_

**¿Por qué un promedio de muchos árboles puede equivocarse menos que el mejor de esos árboles por separado?**

_(tu respuesta)_

## Pregunta de apertura

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

_(tu respuesta)_

## Ejercicio 1: línea base, un árbol y un bosque

Archivo: [`starter/ejercicio1_linea_base_y_ensamblaje.py`](./starter/ejercicio1_linea_base_y_ensamblaje.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), el mismo de la sesión pasada. El script ya trae la preparación de la Semana 9 y una función `evaluar` que entrena cada modelo, mide el error absoluto medio en entrenamiento y en prueba, y lo registra en una tabla. Los 4 `TODO` son los modelos. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-10-sesion-01/starter
python3 ejercicio1_linea_base_y_ensamblaje.py
```

**Respuesta 1 (TODO 1):** la línea base predice siempre la mediana de la tarifa. ¿Cuánto esperas que se equivoque en promedio? ¿Para qué sirve un modelo que no usa ninguna columna?

_(tu respuesta)_

**Respuesta 2 (TODO 2):** un árbol sin límite puede seguir partiendo los datos hasta dejar un viaje por hoja. ¿Qué error esperas en entrenamiento, y qué esperas en prueba?

_(tu respuesta)_

**Respuesta 3 (TODO 3 y 4):** ¿qué esperas que pase con la brecha entre entrenamiento y prueba al limitar la profundidad? ¿Y con 200 árboles promediados?

_(tu respuesta)_

**Lo que confirmó la tabla, TODO por TODO:**

_(anota el error en entrenamiento, en prueba y la brecha de cada modelo)_

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: regularizar una regresión que se sobreajusta

Solo o en pareja. Archivo: [`starter/ejercicio2_regularizacion.py`](./starter/ejercicio2_regularizacion.py), sobre el mismo `viajes_features.csv`. Mismo esqueleto del Ejercicio 1; `evaluar` acepta además `grado_interacciones=3`, que agrega como columnas todos los productos entre columnas hasta grado 3. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

```
python3 ejercicio2_regularizacion.py
```

**Respuesta 1:** ¿cuántas columnas recibe la regresión con interacciones de grado 3, y cuántas filas de entrenamiento hay? ¿Qué tiene que ver esa comparación con el error en prueba que obtuviste?

_(tu respuesta)_

**Respuesta 2:** ¿qué cambió entre `lineal grado 3` y `ridge grado 3` en el error de entrenamiento, y qué cambió en el de prueba? ¿Por qué Ridge empeora un poco en entrenamiento?

_(tu respuesta)_

**Respuesta 3:** ¿qué modelo quedó mejor en prueba? ¿Cuánto mejor que la línea base?

_(tu respuesta)_

**Si terminas antes:** la brecha de la línea base es negativa (se equivoca menos en prueba que en entrenamiento). ¿Cómo puede pasar eso con un modelo que no aprende nada?

_(tu respuesta)_

## Ejercicio 3: los primeros modelos de tu proyecto

Usa tu pipeline de la Semana 9, Sesión 2, con la misma división de entrenamiento y prueba. Esta vez no hay plantilla: escribe el script tú.

Proyecto: predecir el número de emergencias por cantón en t+3, al cierre de t
(`obj_n_emergencias_t3`). Se conserva el pipeline de la Semana 9, Sesión 2:
imputación por mediana y escalado de numéricas, imputación y one-hot del cantón,
y variables de calendario sin cambios. La métrica es MAE, en emergencias por
cantón y día; un valor menor indica mejor predicción.

Esta es una práctica exploratoria de predicción de volumen. El entregable final
del proyecto plantea probabilidades de demanda alta a t+3 y t+7: queda pendiente
calcular el P90 por cantón y día de la semana solo con entrenamiento, construir
las etiquetas y evaluar clasificadores con PR-AUC, precisión y recall frente al
baseline estacional. Los resultados de MAE de esta práctica no evalúan esas alertas.

Los archivos están en la carpeta `proyecto` de esta sesión:
[`modelos_proyecto.py`](./proyecto/modelos_proyecto.py)
y [`resultados_proyecto.csv`](./proyecto/resultados_proyecto.csv).
Desde la raíz del repositorio:

```bash
python3 laboratorios/semana-10-sesion-01/proyecto/modelos_proyecto.py
```

Se usan las mismas 337.120 filas de entrenamiento (2021-07-01 a 2025-08-13)
y 84.448 de prueba (2025-08-17 a 2026-08-28), con 234 columnas preparadas.
Se excluyen 672 filas sin objetivo y 672 filas de separación temporal para que
las etiquetas de entrenamiento ocurran antes del corte del 2025-08-17.
Cada modelo ajusta la preparación exclusivamente con entrenamiento.

### Entregable 1: `modelos_proyecto.py`

Un script, en la carpeta de tu proyecto, que entrene con tu pipeline:

- Una línea base. Elige la que corresponde a tu problema y justifícala abajo: `DummyRegressor` o `DummyClassifier`, o una regla del dominio (por ejemplo, "mañana igual que hoy" si predices una serie en el tiempo).
- Un modelo lineal (`Ridge` si es regresión, `LogisticRegression` si es clasificación), con la regularización que haga falta.
- Un ensamblaje (`RandomForestRegressor`, `GradientBoostingRegressor` o sus versiones `Classifier`).

### Entregable 2: `resultados_proyecto.csv`

Una fila por modelo, con la métrica de tu proyecto en entrenamiento y en prueba, la brecha y los segundos de entrenamiento.

### Entregable 3: respuestas

**¿Cuál es tu línea base, y por qué esa y no otra?**

Uso `DummyRegressor(strategy="median")`, que predice la mediana del objetivo
de entrenamiento para todos los cantones. El objetivo es un conteo numérico,
por lo que corresponde regresión. La mediana minimiza el error absoluto de
una predicción constante y es coherente con MAE. Sirve como referencia mínima
sin utilizar las features; no captura las diferencias entre cantones ni la
estacionalidad. Una regla temporal por cantón sería una referencia adicional
más exigente para una evaluación posterior.

**¿Alguno de tus modelos se sobreajusta? Muestra la evidencia con números de tu tabla.**

El bosque tiene una brecha positiva pequeña: MAE de 5,5056 en entrenamiento
y 5,6328 en prueba, una diferencia de 0,1272 emergencias (2,31% respecto al
error de entrenamiento). Es una señal de un ajuste ligeramente mayor a train,
pero no muestra un sobreajuste fuerte en esta partición temporal. Los modelos
lineales y la línea base tienen brechas negativas; eso puede ocurrir por las
diferencias entre períodos y no implica fuga de información por sí solo.

| Modelo | MAE entrenamiento | MAE prueba | Brecha (prueba − entrenamiento) | Segundos de entrenamiento |
| --- | ---: | ---: | ---: | ---: |
| Línea base mediana | 38,9601 | 36,6301 | −2,3300 | 0,3242 |
| Lineal sin regularizar | 7,7027 | 7,4572 | −0,2455 | 0,3067 |
| Ridge alpha=10 | 7,7014 | 7,3366 | −0,3648 | 0,3004 |
| Bosque regularizado | 5,5056 | 5,6328 | 0,1272 | 95,6787 |

Los tiempos corresponden a esta ejecución en el contenedor y pueden variar.

**¿Qué regularización aplicaste (o por qué no hizo falta), y qué cambió en la tabla?**

Apliqué Ridge con `alpha=10.0`, que penaliza coeficientes grandes. Incluí una
regresión sin penalización como referencia: el MAE de prueba pasó de 7,4572
a 7,3366 y el de entrenamiento de 7,7027 a 7,7014. La mejora en prueba fue
de 0,1206 emergencias, aproximadamente 1,62%. Aquí el MAE de entrenamiento
también bajó ligeramente: Ridge optimiza error cuadrático más penalización,
no MAE, así que no es obligatorio que esta métrica empeore en entrenamiento.

En el bosque limité la profundidad a 12 y exigí al menos 20 observaciones por
hoja, con 100 árboles y `random_state=42`. Su brecha observada fue pequeña
(0,1272). No entrené un bosque sin límites en esta comparación, por lo que
no atribuyo una reducción cuantificada de la brecha a esos límites.
Los parámetros se fijaron antes de evaluar, sin buscar combinaciones con prueba.

**¿Cuánto mejora tu mejor modelo a la línea base?**

El mejor fue el bosque regularizado: redujo el MAE de prueba de 36,6301
a 5,6328 emergencias por cantón y día. La reducción absoluta fue de 30,9973
emergencias y la relativa de 84,62%:
`(36,6301 − 5,6328) / 36,6301 × 100`. Esta mejora corresponde al pronóstico
del volumen a tres días; no mide la detección de demanda alta, que sería
otro objetivo y requeriría construir sus umbrales solo con entrenamiento.

## Antes de la próxima sesión

**Lectura:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

_(tu respuesta)_

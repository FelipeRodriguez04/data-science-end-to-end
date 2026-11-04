# Semana 11, Sesión 1: Gestión y trazabilidad de experimentos: eligiendo el modelo ganador con evidencia

> Objetivo de la sesión: registrar de forma trazable los parámetros, métricas y artefactos de las corridas de entrenamiento de las dos sesiones anteriores, y elegir el modelo final de tu proyecto con esa evidencia.

**Lectura previa de esta sesión:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

**Referencia de esta sesión:** [ML Experiment Tracking (documentación oficial de MLflow)](https://mlflow.org/docs/latest/ml/tracking/).

## Requisitos antes de esta sesión

Hoy se agrega una herramienta: MLflow, que registra cada corrida de entrenamiento. Como en un equipo de datos, el registro vive en un servidor de seguimiento (tracking server), no en tu notebook: el Codespace levanta uno en el puerto 5050 cada vez que inicia, y los notebooks lo encuentran por la variable de entorno `MLFLOW_TRACKING_URI`.

Si tu Codespace se creó antes de esta sesión, sincroniza tu fork y reconstruye el contenedor (`Ctrl+Shift+P` o `Cmd+Shift+P` → "Codespaces: Rebuild Container"; en tu laptop, "Dev Containers: Rebuild Container"). Reconstruir instala MLflow y configura el servidor; tus archivos no se pierden. Al terminar, comprueba desde una terminal:

```
bash .devcontainer/verificar.sh
```

Las últimas líneas deben decir `mlflow, version 2.17.2` y `servidor de MLflow: responde en http://127.0.0.1:5050`. Si el servidor no responde (por ejemplo, después de detener y volver a abrir el Codespace sin que arranque), levántalo a mano con `bash .devcontainer/mlflow-server.sh`.

La interfaz de MLflow se abre desde la pestaña **Ports** de VS Code: el puerto 5050, "MLflow (Semana 11)", con el ícono del globo. Ahí ves cada experimento, sus corridas, y puedes compararlas lado a lado.

Los laboratorios de hoy son notebooks, como los de la Semana 10, y la preparación se importa de `preparacion_taxis.py`. El registro (la base de datos del servidor y los modelos guardados) queda en la carpeta `.mlflow/` de la raíz del repositorio, que no se sube (está en `.gitignore`): lo que entregas es el notebook y la tabla de corridas que exporta.

**Si trabajas en tu laptop sin el contenedor (Opción B del README):** instala con `pip install -r requirements.txt` y, en una terminal aparte desde la raíz de tu fork, levanta el mismo servidor con `mlflow server --backend-store-uri sqlite:///.mlflow/mlflow.db --artifacts-destination .mlflow/artefactos --port 5050`. Antes de abrir los notebooks, define la variable `MLFLOW_TRACKING_URI=http://127.0.0.1:5050`. Sin esa variable, MLflow guarda el registro en una carpeta `mlruns` al lado de cada notebook, sin servidor: los notebooks también funcionan, pero cada carpeta tiene su propio registro.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué tres cosas hay que versionar en un proyecto de machine learning, y con qué herramienta versiona el autor cada una?**

_(tu respuesta)_

**¿Por qué no alcanza con Git para versionar los datos?**

_(tu respuesta)_

## Pregunta de apertura

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

_(tu respuesta)_

## Ejercicio 1: registrar cada corrida con MLflow

Notebook: [`starter/ejercicio1_registrar_corridas.ipynb`](./starter/ejercicio1_registrar_corridas.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Entrena el bosque y el gradient boosting de la Semana 10, y después la corrida de un compañero. Los 4 `TODO` son el registro. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (antes de los TODO):** sin resolver, el notebook entrena los tres modelos e imprime sus errores. ¿Cuál de los dos gradient boosting tiene razón? ¿Qué parte de la salida te permite saberlo, y qué parte no?

Ambos errores son válidos para sus respectivas particiones. La salida muestra MAE de prueba de 1,275352 con semilla 42 y 1,186727 con semilla 7. La semilla identifica que se evaluaron conjuntos diferentes; el menor error por sí solo no demuestra que el segundo modelo sea mejor. Antes de registrar las corridas, los prints tampoco conservaban la huella del CSV ni los hiperparámetros para una auditoría posterior.

**Respuesta 2 (TODO 2 y 3):** de todo lo que se puede registrar de una corrida, ¿qué crees que explica la diferencia entre los dos gradient boosting? ¿Qué parámetro esperas ver distinto en la tabla del paso 5?

La división de los datos: `semilla_division` vale 42 en una corrida y 7 en la otra. Ambos gradient boosting tienen `random_state=42`, 100 estimadores y la misma huella del CSV. Por eso sus MAE no son una comparación sobre las mismas filas de prueba.

**Respuesta 3 (TODO 4):** el registro ya tiene los parámetros y las métricas. ¿Para qué guardar además el modelo entrenado?

Para recuperar el pipeline ya ajustado, incluidos imputadores, escalas, categorías y modelo, sin reentrenarlo. Permite volver a predecir, comprobar la métrica y desplegar exactamente ese artefacto; los parámetros y las métricas solos no contienen los pesos aprendidos.

**Lo que mostró la tabla del paso 5:**

| Modelo | MAE de prueba | Semilla de división | Huella MD5 (12 caracteres) |
|---|---:|---:|---|
| Bosque aleatorio | 1,301589 | 42 | `4800204d50c0` |
| Gradient boosting | 1,275352 | 42 | `4800204d50c0` |
| Gradient boosting (compañero) | 1,186727 | 7 | `4800204d50c0` |

Cada corrida usa 2.380 filas de entrenamiento y 595 de prueba. El experimento conserva también las corridas anteriores; la tabla anterior resume la ejecución completa realizada para este ejercicio.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: comparar con evidencia y elegir

Solo o en pareja. Notebook: [`starter/ejercicio2_comparar_y_elegir.ipynb`](./starter/ejercicio2_comparar_y_elegir.ipynb), sobre el mismo `viajes_features.csv`. Mismo esqueleto del Ejercicio 1, con los cinco modelos de la Semana 10 en un experimento propio. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿qué modelo quedó primero en el registro, y por cuánto le gana al segundo?

La red neuronal quedó primera con MAE de prueba **1,169505**, seguida por gradient boosting con **1,275352**. La diferencia es **0,105847** (aproximadamente 8,30% menos error). Ambas usan las mismas 595 filas de prueba y semilla de división 42. Los cinco controles del notebook imprimen `OK`.

**Respuesta 2:** el paso 5 carga el modelo ganador desde el registro y vuelve a calcular su error. ¿Qué te garantiza que coincida con el número registrado? ¿Qué no te garantiza sobre ese modelo?

Coincidir con tolerancia menor a 1e-9 comprueba que el artefacto cargado reproduce la métrica sobre las mismas filas y con la misma evaluación. No garantiza que generalice a nuevos datos, que sea el mejor en otras divisiones ni que esté libre de sesgos o fuga de información.

**Respuesta 3:** si mañana alguien cambia una fila de `viajes_features.csv` y vuelve a entrenar, ¿cómo lo notarías mirando solo el registro?

Cambiaría `params.huella_datos`: en estas corridas es `4800204d50c0`. La huella detecta cambios de contenido aunque el archivo conserve el nombre; no indica qué fila cambió ni sustituye guardar una versión del dataset.

**Si terminas antes:** abre la interfaz de MLflow (pestaña **Ports**, puerto 5050), entra al experimento `tarifa-taxis-comparacion`, selecciona las cinco corridas y compáralas lado a lado con **Compare**. Además del error, ¿qué te dice el registro de la red neuronal que no te diría una tabla escrita a mano?

El registro conserva el identificador de corrida, los hiperparámetros de la red (capa de 64 neuronas, máximo 1.000 iteraciones y semilla 42), la división, la huella de datos, el tiempo medido, los MAE de entrenamiento y prueba y el pipeline recuperable. La etiqueta `candidato=final` identifica la selección. Una tabla manual podría documentar estos campos, pero no recuperaría por sí sola el artefacto entrenado.

## Ejercicio 3: el registro de tu proyecto

Usa tu `pipeline_proyecto.py` y los modelos que entrenaste en las dos sesiones de la Semana 10. Esta vez no hay plantilla.

Entregables completados junto al pipeline:

- [experimentos_proyecto.ipynb](../semana-09-sesion-02/plantilla-pipeline/experimentos_proyecto.ipynb), ejecutado de principio a fin.
- [corridas_proyecto.csv](../semana-09-sesion-02/plantilla-pipeline/corridas_proyecto.csv), exportado de las cuatro corridas de esa ejecución.

Experimento de MLflow: `ecu911-volumen-emergencias-t3`. Para repetirlo, abre el notebook en el contenedor y usa **Run All** con el servidor de MLflow activo. Cada ejecución crea cuatro corridas nuevas; el CSV incluye exclusivamente las de la ejecución actual.

### Entregable 1: `experimentos_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que importe de ahí la preparación y la división y registre en MLflow, en un experimento con el nombre de tu proyecto, una corrida por cada modelo de la Semana 10 (línea base, modelo lineal, ensamblaje, red neuronal). Cada corrida registra:

- Los hiperparámetros del modelo.
- Cómo se dividieron los datos (la semilla, o la fecha de corte si divides por fecha) y la huella del archivo de datos.
- La métrica de tu proyecto en entrenamiento y en prueba, y los segundos de entrenamiento.
- El pipeline entrenado, como artefacto.

Al final, el notebook marca la corrida que eliges como final (`candidato` = `final`), carga su modelo desde el registro y comprueba que reproduce la métrica registrada.

### Entregable 2: `corridas_proyecto.csv`

La tabla de corridas de tu experimento, exportada con `mlflow.search_runs(...).to_csv("corridas_proyecto.csv", index=False)` desde el mismo notebook. Antes de hacer commit, borra las corridas de prueba del experimento de tu proyecto (en la interfaz de MLflow: selecciónalas y **Delete**), reinicia el kernel y corre todo: la tabla tiene que salir de una corrida completa.

### Entregable 3: la decisión

**¿Qué modelo elegiste como final? Escribe su `run_id` y los números del registro que sostienen la decisión.**

Elegí el **bosque regularizado**, `run_id=0c40681fa7d54d56a75a95cd8ee29f5e`, etiquetado `candidato=final`, porque obtuvo el menor MAE de prueba de los cuatro modelos.

| Modelo | MAE entrenamiento | MAE prueba | Entrenamiento (s) |
|---|---:|---:|---:|
| Bosque regularizado | 5,505628 | **5,632802** | 95,746 |
| Red neuronal | 6,096481 | 5,947196 | 12,113 |
| Ridge (alpha=10) | 7,701378 | 7,336582 | 0,326 |
| Línea base mediana | 38,960139 | 36,630104 | 0,249 |

El bosque mejora en 0,314395 frente a la red y reduce el MAE aproximadamente 84,62% frente a la mediana. Su entrenamiento es más lento: prioricé el menor error. El pipeline recargado produjo MAE **5,632801743210028**, coincidente con el registrado dentro de 1e-9.

Se usaron 337.120 filas de entrenamiento y 84.448 de prueba, con corte **2025-08-17** y separación temporal de tres días para las etiquetas. El MAE está en emergencias por cantón y día. Esta práctica predice volumen a t+3; las alertas de demanda alta a t+3/t+7 siguen requiriendo sus etiquetas y evaluación de clasificación. Elegir sobre este conjunto de prueba convierte su error en evidencia de selección; una estimación final independiente necesitaría otro período reservado.

**Si otra persona clona tu repositorio, ¿qué necesita para obtener exactamente el mismo número? ¿Qué de eso quedó registrado en MLflow y qué no?**

Necesita el mismo CSV, código de preparación, partición temporal, hiperparámetros, semillas y versiones del entorno. La huella SHA-256 del CSV usado es `0a60005c287f3607254b4ded1014146c2deadc793372c75a4bd54b1d2ec34861`.

MLflow guardó la huella, fecha de corte, separación, tamaños de train/test, hiperparámetros (incluidas las semillas), versiones de Python/scikit-learn/pandas/NumPy, métricas, tiempos, el pipeline entrenado y una copia de `pipeline_proyecto.py`. El notebook conserva el procedimiento y `requirements.txt` fija las dependencias del curso. La huella no contiene el dataset: hay que disponer de la misma copia del CSV. Tampoco se fijó una imagen completa del sistema ni el hardware; no se garantiza igualdad bit a bit al reentrenar en otra plataforma, y los tiempos pueden cambiar.

Clonar Git no recupera los modelos ni la base local de MLflow, porque `.mlflow/` está ignorada. Para reproducir mediante el artefacto original hay que conservar o transferir ese registro y sus artefactos; para regenerarlos se ejecuta el notebook con los mismos datos. El CSV exportado conserva la evidencia de esta ejecución, pero no contiene el modelo.

## Antes de la próxima sesión

**Lectura:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

Hoy elegiste el modelo con menor error sobre los 595 viajes de prueba. ¿Cuánto cambiaría ese error si la división hubiera sido otra? ¿Lo viste en algún lado del registro?

En el ejercicio 1, gradient boosting pasó de 1,275352 (semilla 42) a 1,186727 (semilla 7): una diferencia de 0,088625. Se observa junto a `params.semilla_division` en MLflow. Esto ilustra sensibilidad a la partición; no cuantifica todavía la variabilidad de la red ganadora. Para eso harían falta varias particiones o validación cruzada.

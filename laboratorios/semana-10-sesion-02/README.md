# Semana 10, Sesión 2: Entrenamiento y comparación de modelos II: ¿ensamblaje, red neuronal simple o un LLM?

> Objetivo de la sesión: comparar, sobre el problema de tu proyecto, un modelo de ensamblaje, una red neuronal simple entrenada con una librería de alto nivel y un LLM usado vía prompting, y decidir cuál conviene según desempeño, costo y explicabilidad.

**Lectura previa de esta sesión:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

**Referencia de esta sesión:** [Neural network models, supervised (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).

## Requisitos antes de esta sesión

- Nada nuevo que instalar: la red neuronal es `MLPRegressor`, de scikit-learn.
- Acceso a un LLM de chat (ChatGPT, Claude, Gemini u otro; la versión gratuita alcanza), para el Ejercicio 2.
- La tabla de resultados de tu proyecto de la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿para qué tipo de datos y de problema conviene el aprendizaje automático clásico, y para cuál un LLM?**

_(tu respuesta)_

**El artículo compara las dos opciones en datos, costo y explicabilidad. ¿Cuál de esas tres diferencias pesa más para tu proyecto?**

_(tu respuesta)_

## Pregunta de apertura

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

_(tu respuesta)_

## Ejercicio 1: una red neuronal frente al ensamblaje

Archivo: [`starter/ejercicio1_red_neuronal.py`](./starter/ejercicio1_red_neuronal.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv). El script ya entrena el gradient boosting de la sesión pasada con la misma preparación y la misma división. Los 3 `TODO` son la red neuronal, cuántos parámetros aprendió y qué columnas usa más el ensamblaje. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-10-sesion-02/starter
python3 ejercicio1_red_neuronal.py
```

**Respuesta 1 (TODO 1):** sin resolver, la red sale peor que el ensamblaje y aparece una advertencia `ConvergenceWarning`. ¿Qué crees que significa? ¿Qué esperas al dejarla entrenar hasta 1000 iteraciones?

_(tu respuesta)_

**Respuesta 2 (TODO 2):** una capa oculta de 64 neuronas recibe las 25 columnas preparadas y entrega una tarifa. ¿Cuántos números tiene que aprender?

_(tu respuesta)_

**Respuesta 3 (TODO 3):** ¿qué columna esperas que el ensamblaje use más? Si la app le muestra al conductor por qué la tarifa estimada es 70 dólares, ¿cuál de los dos modelos te permite responder?

_(tu respuesta)_

**Lo que confirmó la salida, TODO por TODO:**

_(anota el error y el tiempo de cada modelo, los parámetros de la red y las 3 columnas más importantes)_

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: tres soluciones sobre los mismos 20 viajes

Solo o en pareja. Archivo: [`starter/ejercicio2_tres_soluciones.py`](./starter/ejercicio2_tres_soluciones.py), sobre el mismo `viajes_features.csv`. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación.

1. Completa los `TODO` 1 a 3 y corre el script. Además de entrenar, escribe `prompt_llm.txt`: 30 viajes con su tarifa como ejemplos y 20 viajes sin tarifa.
2. Copia `prompt_llm.txt` completo en un LLM de chat. Guarda los 20 números que responda, uno por línea, en `respuestas_llm.txt`, en la carpeta `starter`.
3. Completa los pasos 4 y 5 y vuelve a correr el script hasta que todas las comprobaciones digan `OK`. Para el paso 5, busca el precio por millón de tokens de entrada del modelo que usaste en la página de precios de su proveedor.

```
python3 ejercicio2_tres_soluciones.py
```

**Respuesta 1:** ¿qué LLM usaste? Completa la tabla con los números de tu corrida.

| | Error en los 20 viajes | Costo de entrenar | Costo de 10 000 viajes al mes | ¿Puedes explicar una predicción? |
|---|---|---|---|---|
| Gradient boosting | _(…)_ | _(…)_ | _(…)_ | _(…)_ |
| Red neuronal | _(…)_ | _(…)_ | _(…)_ | _(…)_ |
| LLM | _(…)_ | _(…)_ | _(…)_ | _(…)_ |

**Respuesta 2:** corre el mismo prompt otra vez, en una conversación nueva. ¿Te dio los mismos 20 números? ¿Qué problema le trae eso a una app que le cobra al pasajero?

_(tu respuesta)_

**Respuesta 3:** busca en tus 20 respuestas la peor predicción del LLM. ¿Qué tiene de particular ese viaje? ¿Lo habría aprendido el ensamblaje con las 2380 filas de entrenamiento?

_(tu respuesta)_

**Si terminas antes:** el error en los 20 viajes de muestra no es el mismo que en los 595 de prueba. ¿Por qué no se mide el LLM sobre los 595, y qué tan confiable es una comparación sobre 20?

_(tu respuesta)_

## Ejercicio 3: ¿qué solución conviene para tu proyecto?

Usa tu pipeline y la tabla de resultados de la sesión pasada. Esta vez no hay plantilla.

En esta práctica seguimos el objetivo exploratorio de la sesión anterior:
predecir el número de emergencias por cantón en t+3, con MAE en unidades de
emergencias por cantón y día. El proyecto final de alertas de demanda alta a
t+3 y t+7 sigue pendiente de construir sus etiquetas con P90 calculado solo
con entrenamiento; los resultados de esta práctica no evalúan esas alertas.

Los entregables están en [`proyecto/`](./proyecto/):
[`comparar_proyecto.py`](./proyecto/comparar_proyecto.py),
[`resultados_comparacion.csv`](./proyecto/resultados_comparacion.csv),
[`evaluacion_10_filas.csv`](./proyecto/evaluacion_10_filas.csv) y
[`prompt_llm_proyecto.txt`](./proyecto/prompt_llm_proyecto.txt).
Para entrenar desde la raíz del repositorio:

```bash
python3 laboratorios/semana-10-sesion-02/proyecto/comparar_proyecto.py
```

Se reutilizan las mismas features, preparación y división temporal del proyecto
de Semana 9: 337.120 filas de entrenamiento y 84.448 de prueba, con 234 columnas
preparadas y una separación de tres días para evitar cruzar etiquetas del corte.
El ensamblaje es el ganador de la sesión anterior: un bosque de 100 árboles,
`max_depth=12`, `min_samples_leaf=20` y `random_state=42`.
La red tiene una capa oculta de 64 neuronas, hasta 300 iteraciones y
`random_state=42`. Se escala el objetivo usando solo entrenamiento y se
revierte la escala al predecir. La parada temprana usa un 10% aleatorio de
entrenamiento para controlar la optimización; el conjunto de prueba temporal
permanece aparte y no decide cuándo detenerla.

La muestra de 10 filas se fija con `random_state=7`. El prompt contiene
30 ejemplos de entrenamiento y esas 10 filas sin su objetivo. Las respuestas
reales del LLM se guardan en `proyecto/respuestas_llm_proyecto.txt`, una por
línea y en orden. Para evaluarlas sin repetir los entrenamientos:

```bash
python3 laboratorios/semana-10-sesion-02/proyecto/comparar_proyecto.py --solo-llm
```

### Entregable 1: `comparar_proyecto.py`

Un script que entrene, con tu pipeline y tu división, el mejor ensamblaje de la sesión pasada y una red neuronal simple (`MLPRegressor` o `MLPClassifier`), y registre para cada uno la métrica de prueba, los segundos de entrenamiento y, para la red, la cantidad de parámetros.

### Entregable 2: la prueba del LLM

Elige 10 filas de tu conjunto de prueba y escribe un prompt con algunas filas de entrenamiento como ejemplos, como el de `prompt_llm.txt`. Pásalo a un LLM y registra su métrica sobre esas 10 filas, junto con la de tus dos modelos sobre las mismas 10. Si tu problema no se puede plantear así (por ejemplo, porque cada predicción necesita una historia larga), explica por qué; eso también es una respuesta.

### Entregable 3: la decisión

Una tabla con las tres opciones (métrica, costo, explicabilidad) y un párrafo que diga cuál usarías en tu proyecto y por qué, con los números de tu tabla.

| Opción | MAE en toda la prueba | MAE en las mismas 10 filas | Entrenamiento | Costo de uso | Explicabilidad |
| --- | ---: | ---: | --- | --- | --- |
| Bosque regularizado | 5,6328 | 6,0975 | 95,11 s | Cómputo local; tarifa monetaria no medida | Permite medir importancias globales de las features; una explicación individual requiere análisis adicional. |
| Red neuronal | 5,9472 | 7,8242 | 11,80 s; 15.105 parámetros | Cómputo local; tarifa monetaria no medida | Los pesos no dan una explicación directa de cada predicción. |
| LLM GPT-6.1 Sol medium | No evaluado en toda la prueba | 5,1000 | Sin entrenamiento local | USD 2,099 de entrada estimados por 10.000 predicciones | Una explicación textual del LLM no demuestra qué determinó su número. |

La red terminó tras 61 iteraciones mediante parada temprana. Los tiempos
corresponden a esta ejecución y pueden variar. Para el LLM, el prompt tiene
4.198 caracteres: con la aproximación de un token por cuatro caracteres,
1.000 prompts de 10 casos y USD 2 por millón de tokens de entrada, el costo
estimado es `4198 / 4 × 1000 × 2 / 1000000 = USD 2,099`. Se usa la
[tarifa API estándar de GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol),
consultada el 2026-10-28. Se excluyen salida y razonamiento de la estimación,
siguiendo la simplificación del ejercicio; no representa el precio de una
suscripción ni una medición del costo de infraestructura local.

Las 10 respuestas del LLM están guardadas en
[`respuestas_llm_proyecto.txt`](./proyecto/respuestas_llm_proyecto.txt).
Su MAE se calculó sobre las mismas 10 filas y en el mismo orden que los otros modelos.

Para esta práctica elegiría el bosque: su MAE de 5,6328
es menor que el 5,9472 de la red, una reducción de aproximadamente 5,29%,
y permite examinar las importancias globales de las variables. La red se
entrenó más rápido (11,80 s frente a 95,11 s), por lo que sería una alternativa
si el tiempo de reentrenamiento fuera prioritario. En las 10 filas ganó
el LLM con MAE de 5,1000, frente a 6,0975 del bosque y 7,8242 de la red.
Ese resultado favorece al LLM en esta muestra, pero 10 casos no permiten
concluir que sea superior en las 84.448 filas de prueba. Prefiero el bosque
porque tiene una evaluación temporal completa y no necesita enviar cada
lote a un proveedor externo; su costo de cómputo local sigue sin medirse.
No comparo directamente el MAE del LLM sobre 10 casos con el del bosque
sobre toda la prueba para declarar un ganador global.
Esta decisión es sobre volumen a t+3; la selección para las alertas finales
deberá repetirse con etiquetas de demanda alta y las métricas del memo.

## Antes de la próxima sesión

**Lectura:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

_(tu respuesta)_

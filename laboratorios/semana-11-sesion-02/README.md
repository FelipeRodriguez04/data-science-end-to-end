# Semana 11, Sesión 2: Esquemas de validación y evaluación de modelos

> Objetivo de la sesión: seleccionar un esquema de validación adecuado al contexto de los datos de tu proyecto (aleatorio, temporal, por grupo) y elegir la métrica de evaluación correcta según el costo relativo de cada tipo de error.

**Lectura previa de esta sesión:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen, capítulo 6 (desarrollo y evaluación de modelos offline).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn. Ten a mano tu `pipeline_proyecto.py` y el modelo final que elegiste la sesión pasada. Los laboratorios de hoy son notebooks, como los de la Semana 10 y la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la paradoja de la exactitud (accuracy paradox)? ¿Qué métricas recomienda en su lugar?**

_(tu respuesta)_

**El artículo cita la ley de Goodhart: "cuando una medida se convierte en objetivo, deja de ser una buena medida". ¿Qué ejemplo da, y qué tiene que ver con elegir un modelo por una sola métrica?**

_(tu respuesta)_

## Pregunta de apertura

La sesión pasada elegiste el modelo con menor error sobre los 595 viajes de prueba. ¿Cuánto cambiaría ese error si la división hubiera sido otra? ¿Lo viste en algún lado del registro?

En la sesión anterior, gradient boosting dio MAE 1,275352 con semilla de división 42 y 1,186727 con semilla 7: una diferencia de 0,088625. MLflow registra ambas junto con la semilla y la misma huella de datos. Eso no mide todavía la variabilidad de la red neuronal ganadora; requiere varias divisiones.

## Ejercicio 1: tres formas de validar el mismo modelo

Notebook: [`starter/ejercicio1_validacion.ipynb`](./starter/ejercicio1_validacion.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Ya trae una función `evaluar` que corre una validación cruzada con el esquema que le pases y guarda el error de cada pliegue. Los 4 `TODO` son tres esquemas de validación y dos métricas. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1):** con una sola división, el gradient boosting dio 1.2754 y el bosque 1.3016. Con 5 pliegues al azar, ¿esperas que los cinco errores del gradient boosting sean parecidos entre sí? ¿Cuánto esperas que varíen?

Esperaría errores relativamente próximos si los viajes fueran intercambiables, aunque una sola división no permite anticipar una dispersión numérica fiable. Al ejecutarlo, gradient boosting obtiene MAE medio 1,2288 y desviación poblacional entre pliegues 0,0361; los cinco errores redondeados son 1,28; 1,24; 1,16; 1,24; 1,23. La variación observada cubre aproximadamente 0,12.

**Respuesta 2 (TODO 2):** en la validación temporal, cada pliegue entrena con los viajes anteriores y valida con los siguientes. ¿Esperas un error mayor o menor que con pliegues al azar? ¿Por qué?

Esperaría mayor error porque se evalúan viajes posteriores, sin mezclar información futura en el entrenamiento, y los primeros pliegues entrenan con menos datos. Se observó MAE medio 1,3099 y desviación 0,0693, frente a 1,2288 y 0,0361 al azar. Esto es compatible con el efecto de período y tamaño de entrenamiento; por sí solo no demuestra deriva.

**Respuesta 3 (TODO 3):** en la validación por zona, ninguna zona de origen aparece a la vez en entrenamiento y en validación. ¿Qué pliegue crees que va a tener el peor error?

Antes de medir, una zona con tarifa especial como JFK Airport es candidata a ser difícil si queda fuera de entrenamiento. El peor observado es el pliegue 5, que retiene Brooklyn y JFK Airport, con MAE 6,43. La media por zona es 2,2996 y la desviación 2,0648. OneHotEncoder ignora las categorías desconocidas, pero eso no permite aprender su efecto específico.

**Respuesta 4 (TODO 4):** la app muestra una tarifa estimada antes del viaje, y el pasajero reclama si la real sale más de 20% por encima. ¿Qué métrica dice mejor cuántos reclamos va a haber: el MAE, el RMSE, o el porcentaje de viajes subestimados en más de 20%?

El porcentaje de viajes cuya tarifa real supera 1,2 veces la estimada refleja directamente el disparador del reclamo descrito. Fue 4,37% (26 de 595 viajes). MAE=1,2754 resume magnitud absoluta y RMSE=1,9213 penaliza más los errores grandes, pero ambos mezclan sobreestimaciones y subestimaciones. No todos los pasajeros afectados necesariamente reclaman: se mide exposición al reclamo, no reclamos observados.

**Lo que confirmó la tabla, esquema por esquema:**

| Esquema / modelo | MAE promedio | Desviación (ddof=0) | Peor pliegue (MAE redondeado) |
|---|---:|---:|---|
| Aleatorio / gradient boosting | 1,2288 | 0,0361 | 1: 1,28 |
| Aleatorio / bosque | 1,2880 | 0,0291 | 4: 1,34 |
| Temporal / gradient boosting | 1,3099 | 0,0693 | 1: 1,40 |
| Por zona / gradient boosting | 2,2996 | 2,0648 | 5: 6,43 |

La tabla exportada está en [validacion_taxis.csv](./starter/validacion_taxis.csv). La desviación describe los cinco pliegues observados, no un intervalo de confianza.

## Durante los ConcepTests

**ConcepTest 1. Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

**ConcepTest 2. Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

## Ejercicio 2: el 95% que no detecta nada

Solo o en pareja. Notebook: [`starter/ejercicio2_fraude_y_umbral.ipynb`](./starter/ejercicio2_fraude_y_umbral.ipynb), sobre [`starter/pagos_taxi.csv`](./starter/pagos_taxi.csv): 6000 pagos de la app de taxis, de los que unos 300 resultaron fraudulentos. Mismo esqueleto del Ejercicio 1, con un problema de clasificación. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿qué accuracy tiene la línea base que nunca predice fraude? ¿Y qué recall?

Accuracy **95,08%** y recall **0%**. Predice legítimos los 6.000 pagos y no detecta ninguno de los 295 fraudes. La precisión de fraude está indefinida porque no predice positivos; scikit-learn la presenta como cero y emite la advertencia correspondiente.

**Respuesta 2:** ¿cuánto mejora la accuracy del gradient boosting sobre la línea base, y cuánto mejora el recall? ¿Cuál de los dos números describe mejor lo que gana la empresa?

Gradient boosting alcanza accuracy 97,12% y recall 46,10%: mejora aproximadamente 2,04 puntos porcentuales en accuracy y 46,10 puntos en recall. Detecta 136 de 295 fraudes con umbral 0,5. El recall muestra mejor la recuperación de fraudes, pero la ganancia empresarial debe evaluarse con el costo de falsos negativos y falsas alarmas.

**Respuesta 3:** con el umbral 0.5, ¿cuántos fraudes deja pasar el gradient boosting? ¿Qué umbral eligió el paso 5, y cuántos fraudes y falsas alarmas tiene con ese umbral?

Con umbral **0,5** deja pasar **159 fraudes** y produce **14 falsas alarmas** (136 fraudes detectados). El mínimo entre los umbrales propuestos es **0,1**: deja pasar **122 fraudes**, detecta **173** y genera **222 falsas alarmas**. Su costo es **$5.546**, frente a **$6.402** con 0,5: ahorro de $856 en estas predicciones fuera de muestra. Los cinco controles muestran `OK`. La selección del umbral y la estimación de su costo usan las mismas predicciones OOF; para estimar el desempeño final sin ese optimismo habría que reservar otro conjunto o anidar la validación.

**Si terminas antes:** si la revisión manual de una falsa alarma costara 20 dólares en vez de 3, ¿el umbral elegido sube o baja? Cambia la constante y comprueba.

Sube de **0,1 a 0,5** entre los umbrales ensayados. Con falsas alarmas de $20, el costo mínimo es **$6.640** con 0,5 (159 fraudes no detectados y 14 falsas alarmas); con 0,1 sería $9.320. Al encarecer la revisión se exige más probabilidad para marcar un pago. Se compararon ambos costos sin alterar la constante original, para conservar la comprobación del ejercicio. Resultados en [costos_fraude.csv](./starter/costos_fraude.csv) y [metricas_fraude.csv](./starter/metricas_fraude.csv).

## Ejercicio 3: la validación y la métrica de tu proyecto

Usa tu `pipeline_proyecto.py` y el modelo final de la sesión pasada. Esta vez no hay plantilla.

Entregables junto al pipeline de Semana 9:

- [validacion_proyecto.ipynb](../semana-09-sesion-02/plantilla-pipeline/validacion_proyecto.ipynb).
- [validacion_proyecto_pliegues.csv](../semana-09-sesion-02/plantilla-pipeline/validacion_proyecto_pliegues.csv): cada pliegue, fechas, tamaños, MAE y costo asimétrico.
- [validacion_proyecto_resumen.csv](../semana-09-sesion-02/plantilla-pipeline/validacion_proyecto_resumen.csv): promedios y desviaciones por esquema y modelo.

Se usan tres pliegues por esquema y los mismos hiperparámetros del bosque y la red de la sesión pasada. Se comparan únicamente sobre el bloque de entrenamiento original (337.120 filas); las 84.448 filas del período de prueba desde 2025-08-17 quedan fuera de esta validación. La preparación se ajusta dentro de cada pliegue.

### Entregable 1: `validacion_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que evalúe tu modelo final (y el segundo mejor) con validación cruzada, con dos esquemas: el que corresponde a tus datos (temporal si predices el futuro, por grupo si la misma entidad aparece muchas veces, estratificado si clasificas una clase poco frecuente) y uno al azar, para comparar. El notebook guarda una tabla con el promedio, la desviación y cada pliegue.

### Entregable 2: la métrica del negocio

Escribe qué error es más caro en tu proyecto (sobreestimar o subestimar, un falso positivo o un falso negativo) y cuánto, aunque sea aproximado. Calcula en el mismo notebook una métrica que refleje ese costo, además de la métrica que venías usando.

### Entregable 3: respuestas

**¿Qué esquema de validación corresponde a tu proyecto, y por qué?**

Corresponde **validación temporal con ventana de entrenamiento creciente**, porque al cierre de t se predice el volumen de emergencias en t+3 para los mismos cantones. Cada pliegue valida fechas posteriores y mantiene juntos todos los cantones del día. Se excluyen tres días entre entrenamiento y validación y se comprueba que la fecha de cada etiqueta de entrenamiento preceda el inicio de validación. El esquema aleatorio permite entrenar con fechas futuras respecto a las evaluadas; validar cantones nuevos respondería otra pregunta de despliegue.

**¿Cambió la conclusión entre el esquema al azar y el tuyo? Muestra los números.**

No cambió el ganador por MAE: el bosque supera a la red en ambos esquemas. Sí cambió el nivel de error y su dispersión:

| Esquema | Modelo | MAE promedio | Desviación MAE (ddof=0) | Peor MAE | Costo asimétrico promedio |
|---|---|---:|---:|---:|---:|
| Aleatorio | Bosque regularizado | 5,8773 | 0,0855 | 5,9798 | 11,7632 |
| Aleatorio | Red neuronal | 6,3596 | 0,1736 | 6,6039 | 12,2408 |
| Temporal | Bosque regularizado | 6,1817 | 0,4651 | 6,8315 | 11,6284 |
| Temporal | Red neuronal | 6,7132 | 0,3795 | 7,2061 | 11,6762 |

El MAE temporal del bosque supera al aleatorio en 0,3044. El temporal refleja la predicción de períodos posteriores y entrena con tamaños crecientes; por eso no se debe atribuir toda la diferencia exclusivamente a fuga temporal. La comparación aleatoria mezcla pasado y futuro y no es la evidencia adecuada de despliegue.

**¿La diferencia entre tu modelo final y el segundo es mayor que la variación entre pliegues? ¿Sigues eligiendo el mismo?**

La diferencia temporal media de MAE es **0,5315** a favor del bosque. Supera ligeramente las desviaciones de cada modelo (0,4651 y 0,3795), pero no el rango completo entre pliegues (1,0629 y 0,9232). El bosque gana por MAE en los tres pares de pliegues; las diferencias red menos bosque son 0,3745; 0,7059; 0,5142. Mantengo el bosque como candidato por MAE, sin afirmar significancia estadística con solo tres períodos. La ventaja en costo asimétrico es apenas 0,0477, menor que la desviación del costo de ambos modelos (aproximadamente 0,6); en esa métrica no hay evidencia de una ventaja estable.

**¿Qué métrica del negocio elegiste, y qué dice que no decía la anterior?**

Usé **error absoluto asimétrico medio**: `media(3 * max(real - predicho, 0) + max(predicho - real, 0))`. La hipótesis de planificación es que subestimar una emergencia cuesta tres veces más que sobreestimarla, por el riesgo de insuficiencia de capacidad. Es una escala de costo relativa propuesta, no una estimación de dólares ni un costo validado con ECU911.

El MAE trata igual una emergencia sobreestimada y una subestimada. La nueva métrica distingue la dirección del error: en validación temporal vale **11,6284** para el bosque y **11,6762** para la red, prácticamente empatados frente a la variación entre períodos. La red tiene menor costo en dos de los tres pliegues, aunque no en el promedio. Antes de una decisión operativa habría que validar el factor 3 con responsables de capacidad y repetir el análisis de sensibilidad. Sigue siendo regresión exploratoria de volumen a t+3; evaluar alertas de demanda alta a t+3/t+7 requiere etiquetas y métricas de clasificación específicas.

## Antes de la próxima sesión

**Lectura:** ["MLOps: Batch vs Online ML Systems"](https://medium.com/mlops-republic/mlops-batch-vs-online-ml-system-cbacee834837), MLOps Republic.

La app de taxis usa dos modelos: uno estima la tarifa antes de pedir el viaje, otro decide si un pago es fraude. ¿Alguno de los dos se podría calcular una vez por noche para todos los casos posibles? ¿Cuál tiene que responder en el momento, y en cuánto tiempo?

_(tu respuesta)_

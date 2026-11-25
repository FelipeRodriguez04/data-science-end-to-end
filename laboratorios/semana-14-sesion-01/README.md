# Semana 14, Sesión 1: Monitoreo en producción II: deriva de datos y deriva conceptual

> Objetivo de la sesión: distinguir deriva de datos de deriva conceptual en el modelo de tu proyecto, y diseñar una alerta que detecte cuál de las dos lo está degradando.

**Lectura previa de esta sesión:** ["Data Drift vs. Concept Drift and Why Monitoring for Them is Important"](https://medium.com/whylabs/data-drift-vs-concept-drift-and-why-monitoring-for-them-is-important-a68c08b141f0), WhyLabs.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Cap. 9: Continual Learning and Test in Production.

## Requisitos antes de esta sesión

Nada nuevo que instalar. El Ejercicio 1 es un notebook (simula la deriva y elige el umbral); el Ejercicio 2, un script del servicio (la alerta queda ahí, corriendo con cada predicción).

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la deriva de datos y qué es la deriva conceptual? Da un ejemplo de cada una.**

La deriva de datos cambia la distribución de las entradas, $P(X)$. Por ejemplo, durante un feriado aumenta la proporción de viajes al aeropuerto y cada tipo de viaje conserva sus reglas de cobro. La deriva conceptual cambia la relación entre entradas y objetivo, $P(Y\mid X)$. Por ejemplo, un alza del taxímetro cambia cuánto cuesta el mismo viaje. [Lectura de WhyLabs](https://medium.com/whylabs/data-drift-vs-concept-drift-and-why-monitoring-for-them-is-important-a68c08b141f0)

**¿Por qué el artículo dice que hay que monitorear las dos, y no solo el desempeño del modelo?**

Las entradas se pueden vigilar desde que llegan y permiten identificar qué variables o segmentos cambiaron. El desempeño requiere conocer el valor real y un deterioro por sí solo deja abierta la causa. Juntar ambas señales ayuda a detectar cambios temprano y a investigar si el error viene de una clientela distinta, de una relación que cambió o de un problema en los datos. [Lectura de WhyLabs](https://medium.com/whylabs/data-drift-vs-concept-drift-and-why-monitoring-for-them-is-important-a68c08b141f0)

## Pregunta de apertura

Desde el 16 de noviembre, el modelo de la tarifa se equivoca casi cuatro veces más que en prueba. Los viajes que recibe son del mismo tipo que los de octubre: las mismas zonas, las mismas distancias, las mismas horas. ¿Qué cambió, entonces? ¿Lo hubieras detectado mirando solo los viajes que llegan?

El aumento de la tarifa por taxímetro hace que viajes equivalentes cuesten más. Es deriva conceptual en el escenario del laboratorio. Las distribuciones de las entradas pueden seguir estables. Para confirmarlo, compararía los cobros reales con las predicciones por tipo de tarifa y comprobaría si el aumento coincide con el cambio de precios.

## Ejercicio 1: ¿deriva de datos o deriva conceptual?

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). [`ejercicio1_deriva.ipynb`](./starter/ejercicio1/ejercicio1_deriva.ipynb) entrena la versión 1 del modelo de la tarifa y simula semanas de 300 viajes: semanas normales, semanas de feriado con más viajes al aeropuerto (deriva de datos) y semanas con un aumento de la tarifa por taxímetro (deriva conceptual). Para cada una mide el PSI de dos features contra los viajes de entrenamiento y el MAE del modelo. Los 4 `TODO`: la fórmula del PSI, medir 200 semanas normales, y las dos derivas simuladas.

Corre todas las celdas antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1 y 2):** antes de medir las semanas normales, ¿qué PSI esperas para una semana en la que nada cambió? Después de medirlas: ¿cuál fue el máximo en 200 semanas?

Esperaría un PSI cercano a cero, con fluctuaciones por tomar solo 300 viajes. En las 200 semanas normales el máximo fue **0,1214** para distancia y **0,1091** para zona.

**Respuesta 2 (TODO 3 y 4):** antes de correr, ¿qué esperas que pase con el PSI y con el MAE en la semana de feriado? ¿Y en la semana del aumento?

En el feriado esperaría que suba el PSI por la mayor presencia de viajes de aeropuerto. El MAE podría variar según el error del modelo en ese tipo de viaje. En la semana del aumento, esperaría el mismo PSI que en la semana normal de la misma semilla y un MAE mayor por la subestimación de las tarifas con taxímetro.

**Respuesta 3 (Paso 4):** en la semana de feriado con peso 8, el PSI es más de diez veces el de una semana normal. ¿Empeoró el modelo? ¿Por qué?

El MAE promedio mejoró de **1,1919** con peso 1 a **0,9440** con peso 8. En esas mismas simulaciones el PSI promedio de distancia subió de 0,0510 a 0,5422 y el de zona de 0,0326 a 0,4193. Aumentó el peso de un grupo que el modelo predice con menor error en esta muestra. Un cambio fuerte en la mezcla de viajes es compatible con una mejora del MAE global.

**Respuesta 4 (Paso 5):** ¿qué umbral de PSI elegirías para la alerta, y con qué dos números de la tabla lo justificas?

Elegiría **0,15**. En la tabla logra **0 %** de falsas alarmas (0 de 200 semanas normales) y **97,5 %** de detección (195 de 200 feriados con peso 4).

**Lo que mostró la simulación:**

| Escenario | PSI de distancia | PSI de zona | MAE |
|---|---|---|---|
| Semanas normales (máximo de 200 por columna) | 0,1214 | 0,1091 | 1,3164 |
| Feriado, aeropuerto ×4 (semilla 1) | 0,1887 | 0,1964 | 0,9694 |
| Aumento de 20% (semilla 1) | 0,0767 | 0,0615 | 4,1638 |

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

Voto razonado para el escenario de apertura: elegiría la opción que identifique **deriva conceptual**. El enunciado y las opciones del ConcepTest no aparecen en el material del laboratorio, así que la letra queda pendiente.

**Tu razonamiento:**

Con viajes comparables, un cambio en la regla de cobro cambia la tarifa esperada. Los cobros y los residuos separados por tipo de tarifa permitirían comprobar la hipótesis.

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

La discusión en pareja queda pendiente. Al revisar la simulación, mantendría la hipótesis para el aumento porque el PSI permanece igual mientras el MAE sube.

## Ejercicio 2: la alerta, en el servicio

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). El servicio de la tarifa de la sesión pasada tiene ahora una ruta `/deriva`. [`deriva.py`](./starter/ejercicio2/deriva.py) trae `MonitorDeDeriva`: guarda los últimos 300 viajes que recibió el servicio y calcula el PSI de `distancia_km` y de `zona_origen` contra la referencia que guarda `entrenar.py` (`referencia_deriva.json`), con el umbral de 0.15. Los pasos 1 a 3 tienen `TODO` (las proporciones numéricas, las categóricas y el PSI); los pasos 4 (guardar solo los últimos viajes) y 5 (el estado, con la alerta) solo tienen un comentario. Resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`. Después:

```
python entrenar.py
python simular_semanas.py
```

[`simular_semanas.py`](./starter/ejercicio2/simular_semanas.py) manda al servicio una semana normal, una de feriado y una del aumento, y muestra `/deriva` después de cada una (no lo modifiques).

**Respuesta 1:** ¿qué dijo `/deriva` después de cada semana? ¿Alguna alerta te sorprendió, o te faltó?

| Semana | Viajes en ventana | PSI de distancia | PSI de zona | Alerta |
|---|---:|---:|---:|---|
| Normal | 300 | 0,0534 | 0,0546 | Apagada |
| Feriado | 300 | 0,2580 | 0,2211 | Encendida |
| Aumento | 300 | 0,0578 | 0,0477 | Apagada |

La alerta se encendió solo en el feriado, como esperaba. Para el aumento hace falta la señal de desempeño porque lo que cambió fue la tarifa cobrada.

**Respuesta 2:** la alerta del monitor no dijo nada en la semana del aumento, que es la que más empeora el modelo. ¿Qué alerta de la sesión pasada sí la habría visto, y por qué tarda más?

La alerta de MAE semanal, una vez que el servicio cruza los cobros con las predicciones mediante `id_prediccion`. Requiere esperar a que terminen los viajes, lleguen los cobros y haya suficientes observaciones para evaluar la semana.

**Respuesta 3:** escribe la regla que usaría el equipo para decidir, con las dos alertas (la de deriva de este ejercicio y la del MAE de la sesión pasada), si lo que pasa es deriva de datos, deriva conceptual o nada. ¿Qué haría en cada caso?

| PSI | MAE con cobros suficientes | Diagnóstico inicial y acción |
|---|---|---|
| Normal | Normal | Sin evidencia en estas señales. Continuar el monitoreo. |
| Alto | Normal | Deriva de datos con desempeño global estable. Revisar calidad de las entradas y errores por segmento. |
| Normal | Alto | Sospecha de deriva conceptual, de una entrada no monitoreada o de un problema con los cobros. Verificar la política tarifaria y los residuos por segmento antes de recalibrar o reentrenar. |
| Alto | Alto | Cambio de entradas con degradación. Revisar si el error creció dentro de cada segmento para distinguir el efecto de la mezcla de un cambio adicional en la relación con la tarifa. Corregir la causa y validar con cobros recientes. |

Con cobros pendientes, el diagnóstico de desempeño queda abierto.

**Si terminas antes:** el monitor mira solo dos features. Inventa una semana con deriva de datos que este monitor no vería. ¿Empeoraría el modelo?

Una semana con más viajes nocturnos y las mismas proporciones de distancias y zonas de origen. El PSI de esas dos entradas se mantendría estable. El MAE podría subir si el modelo se equivoca más de noche. Lo comprobaría comparando el error por franja horaria con los cobros.

## Ejercicio 3: la deriva de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/proyecto/README.md) y el servicio instrumentado de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: la simulación, en `monitoreo/`

Un notebook o un script que, con tus datos de prueba:

- mida el PSI de al menos dos de tus features en ventanas normales (del tamaño de lo que recibe tu servicio en un período), para saber cuánto se mueve sin que nada cambie;
- simule una deriva de datos y una deriva conceptual que tengan sentido en tu problema, y mida el PSI y tu métrica de la Semana 11 en cada una;
- elija el umbral de la alerta con esos números, y guarde la referencia que necesita el servicio.

Para ECU911, la simulación está en [`simular_deriva.py`](../../psets/proyecto/monitoreo/simular_deriva.py) y escribe [`referencia_deriva.json`](../../psets/proyecto/servicio/referencia_deriva.json).

### Entregable 2: la alerta, en el servicio

Un monitor de deriva en tu servicio (puedes adaptar `deriva.py`) con una ruta que diga el PSI de la ventana actual y si hay alerta, y una prueba en `servicio/test_api.py` que compruebe que la alerta se enciende con una ventana alejada de la referencia.

La alerta queda disponible en `GET /deriva`. La [prueba de integración](../../psets/proyecto/servicio/test_api.py) comprueba que se enciende al completar una ventana sesgada y se apaga cuando el tráfico vuelve a la referencia.

### Entregable 3: `docs/monitoreo.md`

Agrega la sección **Deriva**: la tabla de la simulación, el umbral elegido y por qué, y qué deriva no detecta tu alerta.

La sección [Deriva](../../psets/proyecto/docs/monitoreo.md#deriva) documenta los tres escenarios y el umbral de 0,35.

**En tu proyecto, ¿qué deriva de datos es la más probable, y empeoraría tu modelo? ¿Y qué deriva conceptual?**

En ECU911, un cambio en la mezcla de solicitudes que concentre el pronóstico en cantones de alta demanda podría elevar el MAE. Con cuatro veces más peso de muestreo, subió de 5,63 a 10,92 emergencias por cantón y día. Un cambio en el criterio que determina qué llamadas se registran como emergencias podría modificar el conteo objetivo a t+3 con los mismos históricos. En el escenario de +20 %, el MAE subió a 9,77 y el PSI conservó sus valores.

**Si tu alerta de deriva se enciende y tu métrica de desempeño no cambia, ¿qué haces?**

Verifico que haya cobertura suficiente del conteo observado, reviso faltantes y comparo el MAE y el costo de subestimar por cantón. Si el cambio se explica por la mezcla de solicitudes y el desempeño por segmento sigue estable, mantengo el modelo y sigo la señal en los lotes siguientes.

## Antes de la próxima sesión

**Lectura:** ["How to Effectively Communicate Results to Stakeholders As a Data Scientist"](https://medium.com/@sahin.samia/how-to-effectively-communicate-results-to-stakeholders-as-a-data-scientist-711e88dd464f), Sahin Samia.

En las últimas cuatro sesiones construiste un servicio, un registro, métricas y alertas, con PSI, MAE, p95 y recall. ¿Qué cambia entre explicarle tu proyecto a otro científico de datos y a quien paga la factura? Escribe la primera oración que le dirías a cada uno.

- **Científico de datos.** «El bosque estima el volumen de emergencias por cantón a tres días con un MAE de 5,63 en la prueba temporal posterior al corte».
- **Responsable del presupuesto.** «Este pronóstico permite revisar con tres días de antelación qué cantones podrían necesitar más cobertura de atención».

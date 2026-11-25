# Monitoreo del pronóstico de ECU911 a t+3

El servicio predice el número de emergencias para un cantón y una fecha objetivo, tres días después del corte. El conteo observado se puede usar cuando termina ese día y el proceso de datos consolida el cierre. Para esta práctica se espera la observación al día siguiente del objetivo (aproximadamente t+4 desde el corte). Ese plazo es un supuesto de trabajo. Si el conteo aún falta, el reporte deja el pronóstico pendiente y muestra la cobertura de la evaluación.

| Métrica | Tipo | Fuente | Cuándo se conoce | Umbral de alerta propuesto |
|---|---|---|---|---|
| Cantones con predicción por lote | Sistema | Cantones distintos por corte y versión en JSONL, comparados con el catálogo activo | Al cerrar el lote | Menos del 100 % del catálogo esperado 60 min después de que estén disponibles los datos de t |
| Errores de validación (422) | Sistema | `errores_validacion` en `/metricas` y total de peticiones | Al terminar cada petición | Más del 1 % de peticiones en el lote |
| Errores de servidor (5xx) | Sistema | `errores_servidor` en `/metricas` | Al terminar cada petición | Cualquier 5xx en el lote |
| Latencia p50 y p95 | Sistema | Tiempo de respuesta de `POST /predecir`, incluidos errores | Al terminar cada petición | p95 superior a 250 ms con al menos 30 muestras. p50 sirve de referencia |
| Históricos faltantes | Datos, señal indirecta del modelo | Porcentaje de entradas del lote con algún lag o media histórica nulos en el JSONL | En cuanto termina el lote | Más del 5 % de las entradas |
| Cobertura de la verdad | Datos, señal de calidad de evaluación | `evaluados / pronosticos_vencidos` por semana y versión | Tras el cierre esperado de la verdad | Menos del 95 % de los pronósticos ya vencidos con observación disponible |
| MAE semanal | Modelo | Promedio de los errores absolutos al cruzar pronóstico y conteo real | Tras el cierre del día objetivo y la carga del conteo | Mayor que 7,73, aproximadamente 25 % sobre el MAE temporal de referencia (6,1817), en semana completa con al menos 95 % de cobertura |
| Costo asimétrico medio | Modelo | Promedio de 3 × subestimación + sobreestimación en los pronósticos evaluados | Junto con el MAE | Mayor que 14,54, aproximadamente 25 % sobre la referencia temporal de 11,6284, en semana completa con al menos 95 % de cobertura |

Son umbrales iniciales de la práctica, sujetos a revisión con planificación. La API y el reporte exponen las mediciones. Los umbrales de la tabla se aplican al revisar el lote. Para contar errores por lote se toma la diferencia entre lecturas de `/metricas` al inicio y al final del lote en el mismo proceso. Los contadores se reinician con el proceso y los percentiles cubren las últimas 10.000 peticiones. Se usa un solo proceso por contenedor, como en el Dockerfile.

## Registro y reporte

Cada predicción exitosa devuelve un `id_prediccion` único y escribe un evento JSONL con ese ID, momento UTC, versión, entrada validada y salida. La salida incluye cantón, fecha objetivo, corrida y pronóstico. El reporte usa **cantón y fecha objetivo** para unir con una observación única por par. Conserva cada predicción registrada, incluso si hubo varias solicitudes para el mismo par. Los errores se cuentan en `/metricas`. El JSONL contiene únicamente predicciones exitosas.

El CSV de verdad debe incluir `cod_canton` (texto de cuatro dígitos), `fecha` y `n_emergencias`. Debe provenir de un cierre validado con una fila por cantón y día. Desde `psets/proyecto/`:

```bash
python -m monitoreo.desempeno \
  --verdad /ruta/al/conteo_diario.csv \
  --hasta AAAA-MM-DD
```

Sustituye `--hasta` por la última fecha objetivo cuyo conteo debería estar cerrado. Las fechas futuras se excluyen. Los pronósticos vencidos sin observación cuentan en `sin_verdad`. El MAE y el costo se calculan únicamente sobre las filas observadas y quedan sin valor si ninguna tiene conteo. Los períodos empiezan el lunes y se agrupan por fecha objetivo y versión.

Para conservar el registro al recrear el contenedor, reconstruye la imagen con el modelo exportado y monta la carpeta desde `psets/proyecto/`. El puerto 8002 debe estar libre.

```bash
docker build -f servicio/Dockerfile -t servicio-ecu911 .
docker run -d --name servicio-ecu911-monitoreo -p 8002:8000 \
  --mount "type=bind,source=$PWD/monitoreo,target=/app/monitoreo" servicio-ecu911
curl -sS http://localhost:8002/metricas
```

`RUTA_PREDICCIONES` permite cambiar la ubicación del JSONL. Las pruebas usan una carpeta temporal.

## Verificación con datos históricos

Se reprodujeron seis peticiones sobre cortes del 20 y 27 de agosto de 2025, para los cantones 0101, 0901 y 1701, con el bosque registrado como versión 1. Se añadió una entrada inválida para comprobar la instrumentación. En esa ejecución hubo siete peticiones, un 422, ningún 5xx, p50 de 15,390 ms y p95 de 18,681 ms.

| Semana objetivo | Versión | Evaluados | MAE | Costo asimétrico |
|---|---|---:|---:|---:|
| 2025-08-18 | 1 | 3 | 135,4836 | 166,6512 |
| 2025-08-25 | 1 | 3 | 154,0423 | 212,8683 |

Estas cifras corresponden a tres cantones seleccionados y un solo día objetivo por semana. El MAE del conjunto de evaluación de Semana 11 cubre toda la partición temporal. La reproducción demuestra el cruce y el cálculo, y los tiempos corresponden a una ejecución local con TestClient.

La proporción de históricos faltantes permite detectar problemas de disponibilidad de las entradas antes de conocer la verdad. Si el modelo subestima sistemáticamente un cantón con entradas completas, los códigos HTTP, la latencia y esa señal indirecta pueden seguir normales. El error por cantón y el costo asimétrico permiten investigar ese patrón cuando llegan los conteos.

## Deriva

El pronóstico se solicita por cantón al cierre de cada día. Se usó como ventana un lote típico de **224 predicciones** y como referencia la partición de entrenamiento. Para `n_emergencias` y `media_prev_7d` se guardaron los cortes de los deciles y sus proporciones, con una casilla adicional para faltantes. En cada casilla el PSI aplica un mínimo de 0,0001 a las proporciones para calcular el logaritmo.

La simulación usó **377 días de prueba** del 17 de agosto de 2025 al 28 de agosto de 2026, con el bosque exportado como versión 1. Cada día se midió con el lote histórico y con dos perturbaciones. Para simular más solicitudes de cantones de alta demanda, se remuestreó con reemplazo dentro del día, dando peso 4 a los 23 cantones del decil superior por conteo diario medio de entrenamiento. Cada solicitud conservó su conteo real a t+3. Para simular el inicio de un nuevo criterio de registro, se aumentó el conteo objetivo un 20 % y se redondeó a unidades, usando las entradas históricas y los pronósticos del mismo día.

| Escenario | PSI de conteo del día, media / máximo | PSI de media de 7 días, media / máximo | MAE medio (emergencias por cantón y día) | Ventanas con alerta |
|---|---:|---:|---:|---:|
| Histórico | 0,047671 / 0,341947 | 0,038236 / 0,092569 | 5,632802 | 0 de 377 |
| Deriva de datos por mezcla de solicitudes | 0,309946 / 0,908786 | 0,324297 / 0,544014 | 10,915329 | 172 de 377 (45,6 %) |
| Deriva conceptual por conteo objetivo +20 % | 0,047671 / 0,341947 | 0,038236 / 0,092569 | 9,770957 | 0 de 377 |

Se eligió **PSI > 0,35** para cualquiera de las dos variables, redondeando hacia arriba el mayor PSI histórico (0,341947) a la centésima. El máximo de 0,908786 en la mezcla sesgada muestra que parte de esas ventanas se separa del histórico. Este umbral dio cero alertas en los lotes históricos de calibración y detectó 172 de los 377 lotes sesgados. La falsa alarma futura puede variar. La evaluación por día completo tampoco estima la frecuencia en cada posición de la ventana móvil.

`GET /deriva` muestra el PSI calculado sobre las últimas 224 predicciones exitosas del proceso. Mientras se llena la ventana, muestra el PSI disponible con la alerta apagada. La ventana se reinicia al reiniciar el proceso. Al revisar un lote diario, conviene consultar la ruta después de recibir las 224 predicciones.

La prueba HTTP con el modelo exportado dio PSI de **0,052387** y **0,029227** en el lote del 17 de agosto de 2025, con la alerta apagada. Al volver a enviar 224 solicitudes remuestreadas con peso 10 para los conteos del decil superior de ese día, el PSI subió a **1,065848** y **0,799196** y la alerta se encendió.

La alerta deja sin detectar un cambio en el conteo objetivo con entradas idénticas. También puede pasar por alto cambios en otras variables o en la relación entre las dos variables aunque cada distribución marginal se mantenga. El MAE semanal y el costo asimétrico permiten diagnosticar el daño cuando llega la verdad. Si el PSI sube y el MAE se mantiene con cobertura suficiente, se revisa la mezcla de cantones, los faltantes y el error por segmento antes de decidir si hace falta actualizar el modelo.

Desde `psets/proyecto/`, con el modelo exportado y la tabla histórica disponibles:

```bash
python -m monitoreo.simular_deriva
curl -sS http://localhost:8002/deriva
```

El script recalcula `servicio/referencia_deriva.json`. Puede recibir `--datos`, `--modelo` y `--salida` para usar otras rutas. Si regeneras la referencia, reinicia la API o reconstruye la imagen para cargarla. `RUTA_REFERENCIA_DERIVA` permite apuntar a otra referencia al arrancar.

# Despliegue de ECU911: predicción diaria por lotes

El modelo exploratorio estima el volumen de emergencias por cantón en t+3 al cierre de t. El candidato es el bosque regularizado de Semana 11: MAE temporal medio 6,1817; la red obtuvo 6,7132. El proyecto final contempla alertas de demanda alta a t+3 y t+7, que todavía requieren etiquetas, modelos y evaluación propios: este despliegue no debe presentarse como esas alertas ya terminadas.

## Decisión, usuario y latencia

Elegimos **batch diario**. La predicción apoya al equipo de planificación de ECU911 para revisar capacidad, turnos y recursos por cantón con tres días de anticipación. Proponemos como requisito de diseño publicar el lote en **60 minutos** desde el cierre y la disponibilidad de datos del día t. Es una meta por confirmar con el usuario operativo, no una latencia ya medida ni un servicio implementado. No decide el despacho de una emergencia en curso.

## Casos y disponibilidad de datos

La unidad es cantón × fecha objetivo. En los datos usados hay 224 cantones: el modelo actual produciría 224 predicciones por ejecución para t+3. Si se implementan ambos horizontes del proyecto final serían 448 resultados diarios. Las claves se conocen antes del procesamiento y el conjunto de cantones debe verificarse contra el catálogo activo.

Las entradas son el conteo de t, lags de 1/7/14/28 días, medias históricas de 7/28 días, su cociente, código de cantón y calendario de t+3. Los históricos deben cerrarse hasta t sin eventos futuros; el código conserva ceros iniciales. Fin de semana y feriado objetivo se conocen de antemano. La preparación ajustada se reutiliza, nunca se vuelve a ajustar sobre el lote. Su imputación admite faltantes puntuales, pero un lote sin datos diarios o con retraso general debe marcarse como incompleto y generar aviso, no confundirse con ausencia de emergencias.

## Operación propuesta y trazabilidad

Una tarea programada espera el cierre, valida completitud y frescura, construye las features con el mismo código de entrenamiento, carga el pipeline seleccionado y publica una tabla con cantón, fecha de corte, fecha objetivo, horizonte, volumen estimado y versión del modelo. La publicación debe reemplazar el lote de forma atómica y permitir reintentos idempotentes por corte/modelo. Se conserva el modelo anterior y la versión de datos para auditoría. Este documento define el patrón; no afirma que la orquestación o el servicio ya estén desplegados.

## Alternativa online

Con batch se pierde capacidad de actualizar una predicción al llegar un evento durante el día: el resultado queda fijo hasta el lote siguiente. Un endpoint online podría recalcular a petición, pero necesitaría features históricas disponibles con baja latencia, servicio permanente y monitoreo de errores/disponibilidad. Para la planificación diaria y un horizonte de tres días, ese costo no está justificado por el requisito propuesto de una hora. Si el caso cambiara a despacho inmediato o exigiera actualización intradía, habría que revisar el patrón y la definición de las features.

## Pruebas y ejecución

Desde la raíz: `pytest psets/proyecto -v`. Las pruebas fabrican sus DataFrames y un CSV temporal propio; no dependen de los datos reales ni de MLflow. `modelado/pipeline_proyecto.py` contiene la implementación canónica ya evaluada; el archivo de Semana 9 importa esas mismas funciones para mantener compatibles los laboratorios. La API también las importa desde `modelado`.

## Versionado y registro

El modelo registrado se llama **`ecu911-volumen-emergencias-t3`**. El 2026-11-16 se entrenaron y registraron dos versiones, conservando preparación, partición temporal, semilla 42, profundidad 12 y mínimo de 20 filas por hoja:

| Versión | Árboles | MAE de prueba | Run ID | Decisión |
|---|---:|---:|---|---|
| **1** | 100 | **5,632802** | `564e0872f87e4c48801b363c9014425c` | Aprobada como referencia inicial; alias `produccion` |
| 2 | 50 | 5,639604 | `7f62f3997b3846f1993ecb7036abcb05` | Rechazada frente a la 1 |

Cada ejecución de `registrar_modelo` crea una corrida y una versión nuevas; no reemplaza ninguna. El registro conserva el pipeline, métricas, hiperparámetros, huella SHA-256 del CSV, corte temporal y tamaños de los conjuntos. Las dos versiones usaron 337.120 filas de entrenamiento, 84.448 de prueba, corte 2025-08-17 y separación de etiquetas de tres días.

### Criterio y responsable

La promoción se ejecuta explícitamente por el responsable de modelado del proyecto, identificado mediante `--responsable` (en esta práctica: `FelipeRodriguez04`). El script exige datos y evaluación comparables —misma huella de datos, fecha de corte, separación y filas de prueba— y un **MAE estrictamente menor** que el de producción. Sin versión previa permite establecer una referencia. Un empate se rechaza. En cada candidata se guardan las etiquetas `validacion`, `comparada_con`, `responsable`, `motivo`, `fecha_decision_utc` y `tipo_decision`.

Esta es una política educativa sobre el período de evaluación usado en clase; no prueba una mejora estadísticamente significativa ni autoriza por sí sola una operación real. Antes de uso operativo se necesitan aprobación del usuario de planificación, evaluación temporal independiente y revisión de costo asimétrico, calidad/frescura y latencia. Reutilizar este mismo conjunto repetidamente para selección puede sobreajustar decisiones al período.

### Registrar, comparar y exportar

Con MLflow disponible y `MLFLOW_TRACKING_URI=http://127.0.0.1:5050`, desde `psets/proyecto/`:

```bash
python -m servicio.registrar_modelo --arboles 100
python -m servicio.registrar_modelo --arboles 50
# Sustituye VERSION por el número creado; en esta ejecución fueron 1 y 2.
python -m servicio.promover VERSION --responsable FelipeRodriguez04 --motivo "Comparación documentada de MAE y datos"
python -m servicio.exportar_modelo
docker build -f servicio/Dockerfile -t servicio-ecu911 .
docker run -d --name servicio-ecu911-registro -p 8002:8000 servicio-ecu911
```

El exportador consulta una vez el alias `produccion` y carga esa versión por su número fijo, aunque el alias cambie mientras descarga. Guarda `modelo.joblib`, `modelo_version.txt` y `modelo_metadata.json` juntos en `servicio/`, ignorados por Git. Los tres entran a la imagen. La API valida que el número y la huella coincidan y responde `version_modelo` (ahora número de registro, por ejemplo `"1"`), `run_id` y `huella_modelo` en salud y predicción. El contenedor no instala ni consulta MLflow. Mover el alias no modifica un contenedor ya arrancado: hay que volver a exportar, construir y recrear.

### Volver a una versión anterior

1. Identificar la versión que se quiere recuperar en MLflow y documentar el incidente. En esta ejecución la referencia a conservar es la 1.
2. Ejecutar `python -m servicio.promover 1 --rollback --responsable FelipeRodriguez04 --motivo "Restaurar versión 1 tras incidente documentado"`. El rollback es una excepción explícita a la regla de mejora y queda etiquetado con su responsable, fecha y motivo; no elimina versiones.
3. Ejecutar `python -m servicio.exportar_modelo` y reconstruir la imagen.
4. Detener el contenedor anterior y recrearlo desde la nueva imagen. Si se reutiliza un nombre, renombrar o retirar el contenedor detenido antes de ejecutar `docker run`.
5. Verificar `/salud` y una predicción: ambos deben indicar el número restaurado. Conservar la versión problemática para auditar el incidente.

El registro vive en `.mlflow/` y no se incluye en Git. Otro clon necesita recuperar ese registro con sus artefactos o volver a entrenar y registrar; los números de versión de otro registro pueden ser diferentes. La imagen exportada sí puede ejecutar inferencia sin ese servidor.

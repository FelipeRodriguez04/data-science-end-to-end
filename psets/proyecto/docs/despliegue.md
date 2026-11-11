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

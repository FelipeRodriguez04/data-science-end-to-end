# ECU911: planificación de demanda de emergencias

> Este es el README de tu proyecto: reemplaza cada sección entre corchetes a medida que avanzas. Escríbelo para alguien que clona tu fork sin haber estado en clase y necesita entender qué hace el proyecto y correrlo, no como una nota para ti.

El modelo actual estima el volumen de emergencias por cantón en t+3, al cierre de t, como práctica exploratoria para apoyar planificación de capacidad. Las alertas de demanda alta a t+3 y t+7 del proyecto final requieren todavía sus etiquetas y evaluación de clasificación.

**Métrica de éxito:** [la métrica de negocio y el umbral de decisión del PSet 1]

## Datos

| Fuente | Qué contiene | Cómo se obtiene | Dónde queda |
|---|---|---|---|
| [nombre de la fuente] | [qué filas y columnas trae, desde qué fecha] | [la URL, la API o el procedimiento exacto] | `data/raw/[archivo]` |

Los datos no están en el repositorio. Para regenerarlos desde cero: [los comandos, en orden]. Se completa con el PSet 1.

## Cómo correr el pipeline de datos

[Los comandos, en orden y desde qué carpeta, para ir de la fuente al warehouse: levantar la infraestructura, construir la imagen, subir el flow a Kestra, limpiar, cargar y correr dbt. Cada paso dice qué archivo deja y dónde. Se completa con el PSet 2.]

## Cómo entrenar y evaluar el modelo

[Qué notebook corre cada etapa del modelado, en qué orden, y qué necesita levantado (MLflow). Se completa con el PSet 3.]

## Modelo final

[El modelo elegido, su `run_id` en MLflow y la métrica que lo sostiene. El razonamiento completo está en `docs/decision_modelo.md`. Se completa con el PSet 3.]

## Cómo correr el servicio

El servicio usa el bosque regularizado de la corrida MLflow `0c40681fa7d54d56a75a95cd8ee29f5e`, con scikit-learn **1.5.2**. Su MAE en el período de prueba original es **5,632802** y en validación temporal de tres pliegues promedia **6,181659**. Son evaluaciones diferentes. La API predice volumen a t+3, no probabilidad de demanda alta.

### 1. Exportar desde MLflow

Desde la raíz del repositorio, con las dependencias de `requirements.txt` instaladas y el servidor de MLflow activo:

```bash
bash .devcontainer/mlflow-server.sh
cd psets/proyecto
export MLFLOW_TRACKING_URI=http://127.0.0.1:5050
python servicio/exportar_modelo.py
```

Esto carga `runs:/0c40681fa7d54d56a75a95cd8ee29f5e/modelo` y escribe `servicio/modelo.joblib` y `servicio/modelo_metadata.json`. Ambos son locales, están ignorados por Git y se regeneran juntos. El script verifica que la versión de scikit-learn coincida con la registrada y guarda el run ID y la huella SHA-256 del archivo exportado. Al arrancar, la API comprueba esa huella.

El registro `.mlflow/` tampoco se sube a Git. Si clonas en otra máquina, necesitas recuperar sus artefactos o ejecutar el notebook de Semana 11 y exportar la nueva corrida: `python servicio/exportar_modelo.py --run-id ID`. Los identificadores y la huella de esa exportación serán los de tu modelo, no necesariamente los del ejemplo.

### 2. Construir y levantar

Desde `psets/proyecto/`:

```bash
docker build -f servicio/Dockerfile -t servicio-ecu911 .
docker run -d --name servicio-ecu911 -p 8002:8000 servicio-ecu911
```

El contexto debe ser `psets/proyecto/`, porque la imagen copia tanto `modelado/pipeline_proyecto.py` como `servicio/`. El puerto **8002** evita ocupar los puertos 8000 y 8001 usados por los ejercicios de taxis. Abre `/docs` en ese puerto para la documentación interactiva.

Para ejecutarlo directamente, también desde `psets/proyecto/`: `uvicorn servicio.api:app --port 8002` (con el puerto libre). `RUTA_MODELO` permite indicar otro archivo; por defecto carga el que está junto a `api.py`.

### 3. Petición y contrato

El llamador manda la fecha de cierre t, el código de cantón como texto de cuatro dígitos, el conteo del día, sus lags y medias históricas y el indicador de feriado de t+3. Los históricos deben provenir de datos disponibles hasta t y usar las ventanas del entrenamiento. La API no consulta ni calcula el historial desde eventos crudos. Lags y medias faltantes pueden enviarse como `null`; la preparación aprendida maneja sus faltantes. El indicador de feriado se proporciona desde un calendario oficial; la API no lo infiere.

`agregar_features`, importada de `modelado.pipeline_proyecto`, calcula el cociente media de 7 días / media de 28 días y si t+3 cae en fin de semana. La misma implementación es usada por los notebooks de entrenamiento mediante el archivo de compatibilidad de Semana 9. Dentro del pipeline exportado se reutilizan imputación, escalado y codificación ya ajustados; el servicio no entrena ni ajusta de nuevo.

Desde `psets/proyecto/`, usando el ejemplo sintético guardado:

```bash
curl -sS http://localhost:8002/salud
curl -sS -X POST http://localhost:8002/predecir \
  -H 'Content-Type: application/json' \
  --data-binary @servicio/ejemplo.json
```

El JSON de ejemplo es:

```json
{
  "fecha": "2026-11-05",
  "cod_canton": "0101",
  "n_emergencias": 12,
  "lag_1d": 10,
  "lag_7d": 11,
  "lag_14d": 9,
  "lag_28d": 8,
  "media_prev_7d": 10,
  "media_prev_28d": 5,
  "es_feriado_t3": 0
}
```

Respuesta HTTP **200** obtenida con el bosque exportado en el contenedor (entrada sintética de demostración):

```json
{
  "cod_canton": "0101",
  "fecha_corte": "2026-11-05",
  "fecha_objetivo": "2026-11-08",
  "horizonte_dias": 3,
  "emergencias_estimadas": 7.0791,
  "version_modelo": "ea7725843adaa53303713e36a94d12ec612183bef7be61674f54056cb18c55bb",
  "run_id": "0c40681fa7d54d56a75a95cd8ee29f5e"
}
```

`GET /salud` informa si hay modelo cargado y su identidad. La predicción devuelve HTTP 503 si falta el modelo; las entradas inválidas (fecha incorrecta, cantón mal formado, conteos negativos o campos obligatorios ausentes) reciben HTTP 422 antes de llegar al modelo. La salida identifica el corte, objetivo, horizonte, corrida MLflow y versión SHA-256 para trazabilidad. Es una estimación continua de volumen, no un conteo observado ni una garantía de capacidad operativa.

### 4. Pruebas

Desde la raíz del repositorio:

```bash
pytest psets/proyecto -v
```

Hay pruebas del pipeline y del contrato HTTP: salud, predicción conocida con modelo sintético, entradas inválidas rechazadas antes de llamar al modelo, cálculo de features y respuesta sin modelo. No requieren datos privados ni MLflow. El workflow `.github/workflows/pruebas.yml` ya ejecuta este comando en cada push.

FROM python:3.11-slim
WORKDIR /app
COPY servicio/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY modelado/pipeline_proyecto.py ./modelado/pipeline_proyecto.py
COPY servicio/api.py servicio/modelo.joblib servicio/modelo_metadata.json ./servicio/
EXPOSE 8000
CMD ["uvicorn", "servicio.api:app", "--host", "0.0.0.0", "--port", "8000"]

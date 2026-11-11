FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir gunicorn flask spacy langchain-text-splitters
RUN python -m spacy download en_core_web_sm
COPY *.py .
#0.0.0.0:8080 = :8080, timeout 0 = no timeout
ENTRYPOINT ["gunicorn","--bind",":8080","--workers","1","--threads","8","--timeout","0","main:app"]
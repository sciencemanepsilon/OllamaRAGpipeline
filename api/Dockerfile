FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn ollama docling spacy langchain-text-splitters chromadb
RUN python -m spacy download en_core_web_sm
COPY *.pdf *.py .
ENTRYPOINT ["uvicorn", "main:app", "--workers", "1", "--host", "0.0.0.0", "--port", "8000"]
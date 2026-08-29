# SentinelFlow Testbed - Benign Client Node Dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir requests

COPY benign_generator.py /app/benign_generator.py

ENTRYPOINT ["python", "/app/benign_generator.py"]

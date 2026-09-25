FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn jinja2

COPY . /app/frontend/

CMD ["python", "-m", "uvicorn", "frontend.app:app", "--host", "0.0.0.0", "--port", "8000"]

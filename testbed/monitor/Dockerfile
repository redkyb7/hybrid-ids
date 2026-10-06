# SentinelFlow Testbed - IDS Monitor Node Dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install libpcap, tcpdump, and network utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap-dev \
    tcpdump \
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Install CPU ML/DL and network sniffing dependencies
RUN pip install --no-cache-dir \
    scapy \
    numpy \
    pandas==2.2.3 \
    scikit-learn==1.6.1 \
    xgboost==3.4.1 \
    joblib \
    tensorflow-cpu==2.20.0 \
    colorama \
    requests

# Environment settings
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "backend/live_capture.py"]

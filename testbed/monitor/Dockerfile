# SentinelFlow Testbed - IDS Monitor Node Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install libpcap, tcpdump, and network utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap-dev \
    tcpdump \
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Install ML, DL, and network sniffing dependencies
RUN pip install --no-cache-dir \
    scapy \
    numpy \
    pandas \
    scikit-learn \
    xgboost \
    joblib \
    tensorflow \
    colorama \
    requests

# Environment settings
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "backend/live_capture.py"]

# SentinelFlow Testbed - Attacker Node Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install standard penetration testing & packet crafting tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    nmap \
    hping3 \
    hydra \
    curl \
    net-tools \
    iputils-ping \
    tcpdump \
    && rm -rf /var/lib/apt/lists/*

# Install python libraries for attack orchestration
RUN pip install --no-cache-dir requests scapy

COPY attack_campaigns.py /app/attack_campaigns.py

ENTRYPOINT ["python", "/app/attack_campaigns.py", "--attack", "continuous"]

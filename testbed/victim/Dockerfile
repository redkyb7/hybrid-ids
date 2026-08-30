# SentinelFlow Testbed - Victim Node Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install OpenSSH server, curl, and network diagnostic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    curl \
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Configure OpenSSH with user 'admin' and password 'password123'
RUN mkdir -p /var/run/sshd && \
    useradd -rm -d /home/admin -s /bin/bash -u 1000 admin && \
    echo 'admin:password123' | chpasswd && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Install Flask dependencies
RUN pip install --no-cache-dir flask

COPY app.py /app/app.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 80 22

ENTRYPOINT ["/app/entrypoint.sh"]

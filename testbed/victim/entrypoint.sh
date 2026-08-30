#!/bin/bash
set -e

# Start SSH daemon
service ssh start

echo "[Victim Node] OpenSSH Server listening on port 22 (user: admin, password: password123)"
echo "[Victim Node] Starting Web Application on port 80..."

# Start Flask application
exec python /app/app.py

# Dockerfile dla middleware (WebSocket server)
FROM python:3.11-slim

# Ustaw katalog roboczy
WORKDIR /app

# Kopiuj requirements i zainstaluj zależności
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiuj kod aplikacji
COPY . .

# Expose port WebSocket
EXPOSE 3000

# Uruchom serwer WebSocket
CMD ["python", "server.py"]
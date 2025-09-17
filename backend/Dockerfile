# Dockerfile dla serwisu backend (FastAPI)
FROM python:3.11-slim

# Ustaw katalog roboczy
WORKDIR /app

# Kopiuj pliki requirements i kod aplikacji
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port na którym działa FastAPI
EXPOSE 8000

# Komenda startowa dla serwera
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
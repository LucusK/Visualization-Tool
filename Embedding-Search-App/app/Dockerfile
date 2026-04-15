FROM python:3.11-slim

WORKDIR /app

# system deps for pdfplumber (uses pdfminer which needs no extras) and torch CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# create directories that are expected at runtime (volumes will overlay these)
RUN mkdir -p /app/db /app/embeddings /app/output/heatmaps

EXPOSE 5000

CMD ["python", "api.py"]

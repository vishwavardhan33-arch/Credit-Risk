# Build from the project root:   docker build -f docker/Dockerfile -t credit-risk-api .
# Run (model files are mounted, not baked in):
#   docker run --rm -p 8000:8000 -v "$(pwd)/models:/models" credit-risk-api
FROM python:3.11-slim

# LightGBM needs the OpenMP runtime, which slim images do not include
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src ./src

ENV MODEL_DIR=/models \
    PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

RUN useradd --create-home appuser
USER appuser
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

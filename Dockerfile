FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# CPU-only torch keeps the image small; drop --index-url if you want CUDA.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve,llm]"

# Bake backbone weights into the image so containers start offline.
RUN python -c "import timm; timm.create_model('wide_resnet50_2', pretrained=True)"

# Mount a trained model directory at /models/<category> and point MODEL_DIR at it.
ENV MODEL_DIR=/models/bottle
EXPOSE 8000
RUN useradd -m app && chown -R app /app
USER app
CMD ["uvicorn", "industrial_anomaly.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Energy Grid Intelligence - one image, three roles (pipeline job, API, dashboard)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU-only PyTorch keeps the image ~2 GB smaller than the default CUDA wheel
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY app ./app
COPY configs ./configs
COPY tests ./tests
COPY .streamlit ./.streamlit
COPY pyproject.toml README.md ./
# processed data, results and the report are mounted as volumes (see docker-compose.yml)
RUN mkdir -p data/raw data/processed data/external models reports logs

RUN useradd --create-home egi && chown -R egi /app
USER egi

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fs http://localhost:8000/api/v1/health || curl -fs http://localhost:8501/_stcore/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

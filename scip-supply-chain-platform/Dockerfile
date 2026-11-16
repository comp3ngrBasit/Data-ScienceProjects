FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN useradd -m scip && chown -R scip /app
USER scip
EXPOSE 8000 8501
# default: API; the dashboard service overrides the command in docker-compose.yml
CMD ["sh", "scripts/start_api.sh"]

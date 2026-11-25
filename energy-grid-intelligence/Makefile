.PHONY: install pipeline fast api dashboard test lint docker
install:   ; pip install -r requirements.txt
pipeline:  ; python -m app.pipeline run
fast:      ; EGI_FAST=1 python -m app.pipeline run --force
api:       ; uvicorn app.main:app --reload --port 8000
dashboard: ; streamlit run app/dashboard/streamlit_app.py
test:      ; pytest -q
lint:      ; ruff check app tests
docker:    ; docker compose up -d --build db api dashboard

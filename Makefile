.PHONY: install lint format test api ui mlflow docker-up docker-down

install:
	uv sync --group dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest --cov

api:
	uv run uvicorn resumatch.api.main:app --reload --port 8130

ui:
	RESUMATCH_API_URL=http://localhost:8130 uv run streamlit run src/resumatch/ui/app.py --server.port 8631

mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5014

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

build:
	docker compose build

up:
	docker compose up

down:
	docker compose down

test:
	pytest -v

lint:
	python -m ruff check .

format:
	python -m ruff format .

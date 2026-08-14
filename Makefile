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

.PHONY: baseline

baseline:
	./scripts/platform_baseline.sh

.PHONY: smoke

smoke:
	@echo "==> Checking Enterprise AI Platform..."
	@curl --fail --silent --show-error \
		-H "x-api-key: $${API_KEY}" \
		http://localhost:8000/v1/health
	@echo ""
	@echo "==> Checking Prometheus..."
	@curl --fail --silent --show-error \
		http://localhost:9090/-/ready
	@echo ""
	@echo "==> Checking PostgreSQL..."
	@docker compose exec -T postgres pg_isready \
		-U $${POSTGRES_USER:-postgres} \
		-d $${POSTGRES_DB:-vehicle_platform}
	@echo ""
	@echo "==> Docker smoke test passed."

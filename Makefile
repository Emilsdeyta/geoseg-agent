.PHONY: install lint format test typecheck docker-build

install:
	pip install -e ".[dev,api]"
	pre-commit install

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

typecheck:
	mypy src

test:
	pytest -m "not gpu and not slow"

docker-build:
	docker build -f docker/Dockerfile -t geoseg-agent .

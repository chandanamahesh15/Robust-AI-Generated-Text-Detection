.PHONY: help install install-research test lint format train benchmark serve docker clean

help:
	@echo "install          Install core + dev dependencies"
	@echo "install-research Install heavy GPU deps (torch/tf/transformers)"
	@echo "train            Train the deployable TF-IDF model -> artifacts/"
	@echo "benchmark        Reproduce baselines + TF-IDF comparison (CPU)"
	@echo "benchmark-full   Reproduce ALL models (needs research deps + GPU)"
	@echo "serve            Run the FastAPI inference service"
	@echo "docker           Build the slim serving image"
	@echo "test             Run the test suite with coverage"
	@echo "lint             Ruff + mypy static checks"
	@echo "format           Auto-format with ruff"

install:
	pip install -e ".[dev,viz]"

install-research:
	pip install -e ".[research]"

train:
	python -m src.pipeline --config config/config.yaml

benchmark:
	python -m src.benchmark --skip-heavy

benchmark-full:
	python -m src.benchmark

serve:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

docker:
	docker build -t ai-vs-human-detector .

test:
	pytest

lint:
	ruff check src app tests
	mypy src

format:
	ruff format src app tests

clean:
	rm -rf data/extracted/* artifacts/*.joblib __pycache__ .pytest_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

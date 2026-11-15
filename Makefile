.PHONY: all setup test lint format train explain report benchmark clean

VENV := ./venv/bin
PYTHON := $(VENV)/python
PIP := $(VENV)/pip

all: lint test train explain report

setup:
	$(PIP) install -r requirements.txt -r requirements-dev.txt

test:
	$(VENV)/pytest tests/

lint:
	$(VENV)/ruff check src/ tests/ scripts/
	$(VENV)/black --check src/ tests/ scripts/

format:
	$(VENV)/ruff check --fix src/ tests/ scripts/
	$(VENV)/black src/ tests/ scripts/

data:
	$(PYTHON) -m src.data.download_data

train:
	$(PYTHON) scripts/run_all.py --phase train

explain:
	$(PYTHON) scripts/run_all.py --phase explain

benchmark:
	$(PYTHON) scripts/benchmark.py

report:
	$(PYTHON) scripts/run_all.py --phase report

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

.PHONY: all setup test lint format train explain report benchmark app pages clean optimize

VENV := ./venv/bin
PYTHON := $(VENV)/python
PIP := $(VENV)/pip

all: lint test train explain report

setup:
	$(PIP) install -r requirements.txt -r requirements-dev.txt

test:
	$(VENV)/pytest -v tests/

lint:
	$(VENV)/ruff check .
	$(VENV)/black --check .

format:
	$(VENV)/ruff check --fix .
	$(VENV)/black .

data:
	$(PYTHON) -m src.data.download_data

train:
	$(PYTHON) scripts/run_all.py --phase train

explain:
	$(PYTHON) scripts/run_all.py --phase explain

benchmark:
	$(PYTHON) scripts/benchmark.py

app:
	$(VENV)/streamlit run app/app.py

pages:
	$(PYTHON) -m http.server 8000 --directory docs

report:
	$(PYTHON) scripts/run_all.py --phase report

optimize: format lint test
	@echo "All optimization and regression tests passed!"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

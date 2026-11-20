.PHONY: all setup test lint format notebook pages clean optimize push-all verify

VENV := ./venv/bin
PYTHON := $(VENV)/python
PIP := $(VENV)/pip

all: lint test notebook

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

notebook:
	$(VENV)/jupyter nbconvert --to notebook --execute --inplace FinRisk_ML_Group08.ipynb
	cp FinRisk_ML_Group08.ipynb notebooks/FinRisk_ML_Group08.ipynb

pages:
	$(PYTHON) -m http.server 8000 --directory docs

optimize: format lint test
	@echo "All quality and regression tests passed!"

verify:
	$(VENV)/pytest -v tests/

push-all:
	@echo "Pushing to all 4 configured remotes (origin, gitlab, bitbucket, sourceforge)..."
	git push origin main
	git push gitlab main
	git push bitbucket main
	git push sourceforge main

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	rm -rf catboost_info

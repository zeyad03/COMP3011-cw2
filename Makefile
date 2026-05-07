# Convenience targets for the COMP3011-CW2 search engine.
#
# Run `make help` for the list. Most contributors only need `make test`
# and `make typecheck` during the inner loop.

PY ?= python
PIP ?= pip

.PHONY: help install install-dev test test-fast typecheck eval bench docker run clean

help:
	@echo "Targets:"
	@echo "  install       install runtime dependencies"
	@echo "  install-dev   install runtime + dev dependencies"
	@echo "  test          run the full pytest suite with coverage"
	@echo "  test-fast     run the smoke-test subset only"
	@echo "  typecheck     mypy --strict on src/"
	@echo "  eval          run the IR evaluation harness"
	@echo "  bench         run the micro-benchmark script"
	@echo "  docker        build the runtime image"
	@echo "  run           launch the interactive shell"
	@echo "  clean         remove generated caches"

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements-dev.txt

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -q tests/test_smoke.py

typecheck:
	$(PY) -m mypy --strict src/

eval:
	$(PY) evaluation/evaluate.py

bench:
	$(PY) scripts/benchmark.py

docker:
	docker build -t cw2-search .

run:
	$(PY) -m src.main

clean:
	rm -rf .pytest_cache .mypy_cache .hypothesis .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +

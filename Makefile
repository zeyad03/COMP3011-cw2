# Convenience targets for the COMP3011-CW2 search engine.
#
# Run `make help` for the list. Most contributors only need `make test`
# and `make typecheck` during the inner loop.
#
# All targets route through the project-local virtualenv at ./venv,
# so no `source venv/bin/activate` is needed. `make venv` creates it
# on first use; `make shell` drops you into an activated subshell when
# you want to run ad-hoc commands by hand.

VENV ?= venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help venv shell install install-dev test test-fast typecheck eval bench docker run clean

help:
	@echo "Targets:"
	@echo "  venv          create the local virtualenv at ./$(VENV) (idempotent)"
	@echo "  shell         launch a subshell with the venv activated"
	@echo "  install       install runtime dependencies into the venv"
	@echo "  install-dev   install runtime + dev dependencies into the venv"
	@echo "  test          run the full pytest suite with coverage"
	@echo "  test-fast     run the smoke-test subset only"
	@echo "  typecheck     mypy --strict on src/"
	@echo "  eval          run the IR evaluation harness"
	@echo "  bench         run the micro-benchmark script"
	@echo "  docker        build the runtime image"
	@echo "  run           launch the interactive shell"
	@echo "  clean         remove generated caches"

# Create the venv on first use; idempotent thereafter. All other targets
# depend on this so a fresh clone can run `make test` directly.
$(VENV)/bin/python:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip

venv: $(VENV)/bin/python

# Drop into an interactive subshell with the venv activated. `exit`
# returns to the parent shell. Works around the fundamental limit that
# a Make recipe can't mutate the parent shell's environment.
shell: venv
	@echo "Entering subshell with $(VENV) activated. Type 'exit' to leave."
	@$$SHELL -c '. $(VENV)/bin/activate && exec $$SHELL'

install: venv
	$(PIP) install -r requirements.txt

install-dev: venv
	$(PIP) install -r requirements-dev.txt

test: venv
	$(PY) -m pytest

test-fast: venv
	$(PY) -m pytest -q tests/test_smoke.py

typecheck: venv
	$(PY) -m mypy --strict src/

eval: venv
	$(PY) evaluation/evaluate.py

bench: venv
	$(PY) scripts/benchmark.py

docker:
	docker build -t cw2-search .

run: venv
	$(PY) -m src.main

clean:
	rm -rf .pytest_cache .mypy_cache .hypothesis .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +

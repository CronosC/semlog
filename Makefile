.PHONY: test test-fast coverage format lint typecheck check install example build-all

RUN := uv run

install:
	uv sync --extra dev
	$(RUN) pre-commit install

test:
	$(RUN) pytest -n auto

# Skips the tests that start a JVM or a subprocess.
test-fast:
	$(RUN) pytest -n auto -m "not reasoner and not slow"

coverage:
	$(RUN) pytest -n auto --cov=semanticlog --cov-report=html --cov-report=term-missing

format:
	$(RUN) ruff format semanticlog/ tests/ benchmarks/
	$(RUN) ruff check --fix semanticlog/ tests/ benchmarks/

lint:
	$(RUN) ruff check semanticlog/ tests/ benchmarks/
	$(RUN) ruff format --check semanticlog/ tests/ benchmarks/

typecheck:
	$(RUN) mypy semanticlog/

check: lint typecheck test

# Build the committed example. Needs no data beyond what is in the repo.
example:
	$(RUN) semlog build --config examples/pipeline.toml

# Build every run config. runs/ is local and gitignored, and _base.toml holds
# the settings the others extend.
build-all:
	@test -d runs || { echo "no runs/ directory. Start from examples/pipeline.toml"; exit 2; }
	@for config in runs/[!_]*.toml; do \
		test -e "$$config" || continue; \
		echo "=== $$config ==="; \
		$(RUN) semlog build --config "$$config" || exit 1; \
	done

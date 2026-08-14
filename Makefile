.PHONY: test test-fast coverage format lint typecheck check install example build build-all

# Every tool comes from the project environment, not from whatever the shell
# happens to have on PATH.
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

# Build one run config: make build CONFIG=runs/hiring.toml
CONFIG ?=
build:
	@test -n "$(CONFIG)" || { echo "usage: make build CONFIG=runs/<name>.toml"; exit 2; }
	$(RUN) semlog build --config $(CONFIG)

# Build every run config. _base.toml holds shared settings.
build-all:
	@test -d runs || { \
		echo "no runs/ directory. Run configs are local and gitignored;"; \
		echo "start from examples/pipeline.toml, or use: make example"; \
		exit 2; \
	}
	@found=0; for config in runs/[!_]*.toml; do \
		test -e "$$config" || continue; \
		found=1; \
		echo "=== $$config ==="; \
		$(RUN) semlog build --config "$$config" || exit 1; \
	done; \
	test "$$found" = 1 || { echo "runs/ holds no run configs (only _base.toml?)"; exit 2; }

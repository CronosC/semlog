.PHONY: test coverage format lint typecheck check install example build build-all

install:
	uv sync --extra dev
	pre-commit install

test:
	pytest tests/

coverage:
	pytest --cov=semanticlog --cov-report=html --cov-report=term-missing tests/

format:
	ruff format semanticlog/ tests/
	ruff check --fix semanticlog/ tests/

lint:
	ruff check semanticlog/ tests/

typecheck:
	mypy semanticlog/

check: format lint typecheck test

# Build the committed example. Needs no data beyond what is in the repo.
example:
	semlog build --config examples/pipeline.toml

# Build one run config: make build CONFIG=runs/hiring.toml
CONFIG ?=
build:
	@test -n "$(CONFIG)" || { echo "usage: make build CONFIG=runs/<name>.toml"; exit 2; }
	semlog build --config $(CONFIG)

# Build every run config. _base.toml is shared settings, not a run of its own.
build-all:
	@for config in runs/[!_]*.toml; do \
		echo "=== $$config ==="; \
		semlog build --config "$$config" || exit 1; \
	done

.PHONY: help test test-fast coverage format lint typecheck check verify install example build-all

RUN := uv run

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | expand -t 12

install: ## Install the dependencies and the git hooks
	uv sync --locked --extra dev
	$(RUN) pre-commit install

test: ## Run the whole suite
	$(RUN) pytest -n auto

test-fast: ## Run the suite without the JVM and subprocess tests
	$(RUN) pytest -n auto -m "not reasoner and not slow"

coverage: ## Run the suite with an HTML coverage report
	$(RUN) pytest -n auto --cov=semanticlog --cov-report=html --cov-report=term-missing

format: ## Format and fix the code
	$(RUN) ruff format semanticlog/ tests/ benchmarks/
	$(RUN) ruff check --fix semanticlog/ tests/ benchmarks/

lint: ## Lint and check formatting
	$(RUN) ruff check semanticlog/ tests/ benchmarks/
	$(RUN) ruff format --check semanticlog/ tests/ benchmarks/

typecheck: ## Type check the package
	$(RUN) mypy semanticlog/

check: lint typecheck test ## Lint, type check and run the whole suite

verify: check example ## Check, build examples/pipeline.toml and test it for consistency under HermiT
	$(RUN) semlog reason examples/out/pipeline.ofn --check

example: ## Build examples/pipeline.toml
	$(RUN) semlog build --config examples/pipeline.toml

build-all: ## Build every config in datasets/runs/ not starting with _
	@set -- datasets/runs/[!_]*.toml; [ -e "$$1" ] || { echo "No config in datasets/runs/. Start from examples/pipeline.toml." >&2; exit 2; }; scripts/build_runs.sh "$$@"

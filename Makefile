.PHONY: help test test-fast coverage format lint typecheck check verify install example build-all

RUN := uv run

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | expand -t 12

install: ## Install the dependencies and the git hooks
	uv sync --extra dev
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

verify: check ## Check, build examples/pipeline.toml and test it for consistency under HermiT
	$(RUN) semlog build --config examples/pipeline.toml
	$(RUN) semlog reason examples/out/pipeline.ofn --check --backend HermiT

example: ## Build examples/pipeline.toml
	$(RUN) semlog build --config examples/pipeline.toml

# Configs starting with _ are skipped. They hold settings the others extend.
# A config whose declare_file does not exist is mined first.
# A failed config does not stop the rest. The failures are summarized at the end.
# Ctrl-C stops the current config and ends the run. The summary is still printed.
build-all: ## Build every config in runs/
	@test -d runs || { echo "No runs/ directory. Start from examples/pipeline.toml."; exit 2; }
	@work=$$(mktemp -d); trap 'rm -rf "$$work"' EXIT; trap ':' INT; \
	failed=""; \
	for config in runs/[!_]*.toml; do \
		test -e "$$config" || continue; \
		name=$$(basename "$$config" .toml); \
		log="$$work/$$name.log"; \
		echo "=== $$config ==="; \
		declare_file=$$($(RUN) semlog build --config "$$config" --dry-run 2>/dev/null \
			| sed -n 's/^declare_file = "\(.*\)"$$/\1/p'); \
		if test -n "$$declare_file" && ! test -f "$$declare_file"; then \
			echo "--- mining $$declare_file ---"; \
			( trap ':' INT; $(RUN) semlog mine-declare --config "$$config" 2>&1; echo $$? >"$$work/status" ) \
				| tee "$$log"; \
			status=$$(cat "$$work/status"); \
			if test "$$status" -ne 0; then \
				echo "MINING FAILED $$config (exit $$status)"; \
				failed="$$failed $$name"; \
				test "$$status" -ne 130 || break; \
				continue; \
			fi; \
		fi; \
		( trap ':' INT; $(RUN) semlog build --config "$$config" 2>&1; echo $$? >"$$work/status" ) \
			| tee "$$log"; \
		status=$$(cat "$$work/status"); \
		test "$$status" -eq 0 && continue; \
		echo "FAILED $$config (exit $$status)"; \
		failed="$$failed $$name"; \
		test "$$status" -ne 130 || break; \
	done; \
	test -n "$$failed" || exit 0; \
	echo; \
	echo "=== failed ==="; \
	for name in $$failed; do \
		echo "--- runs/$$name.toml ---"; \
		tail -n 15 "$$work/$$name.log"; \
		echo; \
	done; \
	exit 1

.PHONY: test coverage format lint typecheck check install build-onto build-ocel

# build-onto: general OCEL→ontology builder (set ONTO_FOLDER, ONTO_FILES, BUILD_* vars to customise)
ONTO_FOLDER   ?=
ONTO_FILES    ?=
BUILD_PROFILE ?= OWL2DL
BUILD_ORDER   ?= DF
BUILD_EXTRA   ?= --data-representation equal_bins --materialize

build-onto:
	python scripts/build_onto.py \
		--folder $(ONTO_FOLDER) \
		$(if $(ONTO_FILES),--files $(ONTO_FILES),) \
		--profile $(BUILD_PROFILE) \
		$(if $(BUILD_ORDER),--order $(BUILD_ORDER),--order) \
		$(BUILD_EXTRA)

# if empty then all ocel files are build!
OCEL_FILES ?=

build-ocel:
	$(MAKE) build-onto \
		ONTO_FOLDER=data/ocel \
		ONTO_FILES="$(OCEL_FILES)" \
		BUILD_PROFILE=OWL2DL \
		BUILD_ORDER=DF \
		BUILD_EXTRA="--data-representation equal_bins --materialize --ocel-extensions ATTRIBUTE_HISTORY --size-limit 100"

install:
	uv sync
	pre-commit install

test:
	pytest tests/

coverage:
	pytest --cov=semanticlog --cov-report=html --cov-report=term-missing tests/

format:
	ruff format semanticlog/ scripts/ tests/
	ruff check --fix semanticlog/ scripts/ tests/

lint:
	ruff check semanticlog/ scripts/ tests/

typecheck:
	mypy semanticlog/

check: format lint typecheck test

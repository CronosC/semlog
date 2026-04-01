.PHONY: test coverage format lint typecheck check install build-onto build-ocel

# build-onto: general OCEL→ontology builder (set ONTO_FOLDER, ONTO_FILES, BUILD_* vars to customise)
ONTO_FOLDER   ?=
ONTO_FILES    ?=
BUILD_PROFILE ?= OWL2DL
BUILD_ORDER   ?= DF
BUILD_EXTRA   ?= --quantize --materialize

build-onto:
	python scripts/build_onto.py \
		--folder $(ONTO_FOLDER) \
		$(if $(ONTO_FILES),--files $(ONTO_FILES),) \
		--profile $(BUILD_PROFILE) \
		$(if $(BUILD_ORDER),--order $(BUILD_ORDER),--order) \
		$(BUILD_EXTRA)

OCEL_FILES ?=

build-ocel:
	$(MAKE) build-onto \
		ONTO_FOLDER=data/ocel \
		ONTO_FILES="$(OCEL_FILES)" \
		BUILD_PROFILE=OWL2DL \
		BUILD_ORDER=DF \
		BUILD_EXTRA="--quantize --materialize"

install:
	pip install -e .
	pre-commit install

test:
	pytest tests/

coverage:
	pytest --cov=semanticlog --cov-report=html --cov-report=term-missing tests/

format:
	pre-commit run --all-files

lint:
	flake8 semanticlog/ scripts/ tests/

typecheck:
	mypy semanticlog/

check: format lint typecheck test

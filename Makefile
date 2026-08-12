.PHONY: test coverage format lint typecheck check install example build build-all data_download data_upload data_delete

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
	ruff format --check semanticlog/ tests/

typecheck:
	mypy semanticlog/

check: lint typecheck test

# Build the committed example. Needs no data beyond what is in the repo.
example:
	semlog build --config examples/pipeline.toml

# Build one run config: make build CONFIG=runs/hiring.toml
CONFIG ?=
build:
	@test -n "$(CONFIG)" || { echo "usage: make build CONFIG=runs/<name>.toml"; exit 2; }
	semlog build --config $(CONFIG)

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
		semlog build --config "$$config" || exit 1; \
	done; \
	test "$$found" = 1 || { echo "runs/ holds no run configs (only _base.toml?)"; exit 2; }

# Optional datasets. data/ is gitignored, so the datasets live as assets on a
# GitHub release instead of in the repo:
#
#   make data_download GROUP=ocel   # anyone, no token needed
#   make data_upload   GROUP=ocel   # maintainer, needs GITHUB_TOKEN
#   make data_delete   GROUP=ocel   # maintainer, removes the uploaded assets
#
# Each name below is one file or directory under data/<group>/ and becomes one
# <group>-<name>.tar.zst asset. Names cannot contain spaces, because make splits
# the list on whitespace.
DATA_REPO ?= CronosC/semlog
DATA_TAG  ?= datasets-v1
DATA_WORK ?= datasets/.release

# Which datasets each group publishes. Fill these in, or pass the list on the
# command line:
#
#   make data_download GROUP=ocel DATASETS_ocel="P2P.xml Hiring.xml"
DATASETS_declare ?=
DATASETS_ocel    ?=

GROUP ?=
DATASETS = $(DATASETS_$(GROUP))

# Shared preconditions for the three data targets.
define _require_group
	@test -n "$(GROUP)" || { echo "usage: $(1) GROUP=<declare|ocel>"; exit 2; }
endef

define _require_datasets
	@test -n "$(DATASETS)" || { \
		echo "GROUP=$(GROUP) publishes nothing yet."; \
		echo "Set DATASETS_$(GROUP) in the Makefile, or pass it:"; \
		echo "  $(1) GROUP=$(GROUP) DATASETS_$(GROUP)=\"a.xml b.json\""; \
		exit 2; \
	}
endef

data_download:
	$(call _require_group,make data_download)
	$(call _require_datasets,make data_download)
	@mkdir -p datasets/$(GROUP) $(DATA_WORK)
	@for name in $(DATASETS); do \
		asset="$(GROUP)-$$name.tar.zst"; \
		echo "=== $$asset ==="; \
		curl -fL --progress-bar -o "$(DATA_WORK)/$$asset" \
			"https://github.com/$(DATA_REPO)/releases/download/$(DATA_TAG)/$$asset" || exit 1; \
		tar -C datasets/$(GROUP) -x --zstd -f "$(DATA_WORK)/$$asset" || exit 1; \
	done

# Uploading the same asset name twice is a 422 from the API. To republish,
# bump DATA_TAG or run data_delete first.
data_upload:
	$(call _require_group,make data_upload)
	$(call _require_datasets,make data_upload)
	@test -n "$$GITHUB_TOKEN" || { echo "GITHUB_TOKEN is not set"; exit 2; }
	@mkdir -p $(DATA_WORK)
	@release_id=$$(curl -fsSL -H "Authorization: Bearer $$GITHUB_TOKEN" \
		"https://api.github.com/repos/$(DATA_REPO)/releases/tags/$(DATA_TAG)" \
		| python3 -c "import json,sys; print(json.load(sys.stdin)['id'])") || \
		{ echo "no release tagged $(DATA_TAG) in $(DATA_REPO). Create it first."; exit 1; }; \
	for name in $(DATASETS); do \
		asset="$(GROUP)-$$name.tar.zst"; \
		echo "=== $$asset ==="; \
		tar -C datasets/$(GROUP) -c --zstd -f "$(DATA_WORK)/$$asset" "$$name" || exit 1; \
		curl -fL --progress-bar -X POST -o /dev/null \
			-H "Authorization: Bearer $$GITHUB_TOKEN" \
			-H "Content-Type: application/octet-stream" \
			--data-binary "@$(DATA_WORK)/$$asset" \
			"https://uploads.github.com/repos/$(DATA_REPO)/releases/$$release_id/assets?name=$$asset" \
			|| exit 1; \
	done

# Deletes the release assets named in DATASETS_<group>. The release itself and
# the local copies under data/<group>/ are left alone. Deletion is permanent.
data_delete:
	$(call _require_group,make data_delete)
	$(call _require_datasets,make data_delete)
	@test -n "$$GITHUB_TOKEN" || { echo "GITHUB_TOKEN is not set"; exit 2; }
	@mkdir -p $(DATA_WORK)
	@curl -fsSL -o "$(DATA_WORK)/release.json" -H "Authorization: Bearer $$GITHUB_TOKEN" \
		"https://api.github.com/repos/$(DATA_REPO)/releases/tags/$(DATA_TAG)" || \
		{ echo "no release tagged $(DATA_TAG) in $(DATA_REPO)"; exit 1; }
	@for name in $(DATASETS); do \
		asset="$(GROUP)-$$name.tar.zst"; \
		id=$$(python3 -c "import json,sys; m=[a for a in json.load(open(sys.argv[1]))['assets'] if a['name']==sys.argv[2]]; print(m[0]['id'] if m else '')" \
			"$(DATA_WORK)/release.json" "$$asset") || exit 1; \
		if [ -z "$$id" ]; then echo "=== $$asset (not on $(DATA_TAG), skipped)"; continue; fi; \
		echo "=== $$asset (deleting)"; \
		curl -fsSL -X DELETE -o /dev/null -H "Authorization: Bearer $$GITHUB_TOKEN" \
			"https://api.github.com/repos/$(DATA_REPO)/releases/assets/$$id" || exit 1; \
	done

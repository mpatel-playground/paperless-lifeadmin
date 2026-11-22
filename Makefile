# Load .env if present so OLLAMA_MODEL etc. are available to recipes.
-include .env
export

COMPOSE := docker compose -f docker-compose.dev.yml
OLLAMA_MODEL ?= qwen3.5:9b
VENV := .venv
BIN := $(VENV)/bin

.PHONY: up up-llm down logs pull-model install lint fmt test check

up: ## Start Paperless + broker
	$(COMPOSE) up -d

up-llm: ## Start Paperless + broker + Ollama in Docker
	$(COMPOSE) --profile llm up -d

down: ## Stop all services (data in .devdata/ is kept)
	$(COMPOSE) --profile llm down

logs: ## Follow logs from all running services
	$(COMPOSE) --profile llm logs -f

pull-model: ## Pull OLLAMA_MODEL into the native Ollama install
	@command -v ollama >/dev/null || { echo "ollama not found: install it from https://ollama.com/download"; exit 1; }
	ollama pull $(OLLAMA_MODEL)

install: ## Create .venv with Python 3.12 and install the package with dev tools
	python3.12 -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -e ".[dev]"

lint: ## Lint and check formatting (what CI runs)
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

fmt: ## Auto-fix lint issues and format code
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

test: ## Run the test suite
	$(BIN)/pytest

check: lint test ## Everything that must pass before a commit

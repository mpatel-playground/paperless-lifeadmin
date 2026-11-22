# Load .env if present so OLLAMA_MODEL etc. are available to recipes.
-include .env
export

COMPOSE := docker compose -f docker-compose.dev.yml
OLLAMA_MODEL ?= qwen3.5:9b

.PHONY: up up-llm down logs pull-model

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

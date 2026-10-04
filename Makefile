# RAGScope shortcuts. Each target is a plain command, listed in README.md for shells
# without make (e.g. PowerShell).
.PHONY: up down logs test test-neo4j lint fe-dev fe-check schemas samples

up: ## build and start backend, frontend and Neo4j
	docker compose up --build -d

down: ## stop the stack (data volumes are kept)
	docker compose down

logs:
	docker compose logs -f backend frontend

test: ## backend tests in Docker
	docker compose run --rm --no-deps backend python -m pytest -q

test-neo4j: ## backend tests including the Neo4j store tests
	docker compose up -d neo4j
	docker compose run --rm -e NEO4J_TEST_URI=bolt://neo4j:7687 -e NEO4J_TEST_PASSWORD=graphrag-dev backend python -m pytest -q

lint: ## ruff (backend, in Docker) and ESLint + typecheck (frontend)
	docker compose run --rm --no-deps backend sh -c "python -m ruff check app tests && python -m ruff format --check app tests"
	cd frontend && npm run lint && npm run typecheck

fe-dev: ## frontend dev server on the host, against the backend on :8010
	cd frontend && NEXT_PUBLIC_API_URL=http://localhost:8010 npm run dev

fe-check: ## frontend unit tests
	cd frontend && npm test

schemas: ## regenerate frontend zod schemas from the backend models
	cd frontend && npm run gen:schemas

samples: ## regenerate the PDF/DOCX sample documents
	docker compose run --rm --no-deps -v "./samples:/samples" backend python /samples/build_samples.py

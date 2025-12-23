.PHONY: help build up down logs test clean

help:
	@echo "Available commands:"
	@echo "  make build    - Build Docker images"
	@echo "  make up       - Start services"
	@echo "  make down     - Stop services"
	@echo "  make logs     - View logs"
	@echo "  make test     - Run tests"
	@echo "  make clean    - Clean up volumes and data"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

test:
	pytest tests/

clean:
	docker-compose down -v
	rm -rf data/ qdrant_storage/


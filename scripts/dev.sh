#!/bin/bash
# Development helper script for rag-chatbot

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

function show_help() {
    echo "RAG Chatbot - Development Helper"
    echo ""
    echo "Usage: ./scripts/dev.sh <command>"
    echo ""
    echo "Commands:"
    echo "  install       Install dependencies in development mode"
    echo "  lint          Run linting (ruff, mypy)"
    echo "  format        Format code with ruff"
    echo "  test          Run tests with coverage"
    echo "  api           Start API server"
    echo "  ui            Start Streamlit UI"
    echo "  ollama        Start Ollama and pull models"
    echo "  docker-up     Start full stack with docker-compose"
    echo "  docker-down   Stop docker-compose stack"
    echo "  docker-logs   Show docker-compose logs"
    echo "  clean         Clean build artifacts and cache"
    echo "  help          Show this help"
}

function install_deps() {
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -e ".[dev,ui]"
    pre-commit install
}

function run_lint() {
    echo "Running linters..."
    ruff check .
    mypy app
}

function run_format() {
    echo "Formatting code..."
    ruff format .
    ruff check --fix .
}

function run_tests() {
    echo "Running tests..."
    pytest --cov=app --cov-report=term-missing --cov-report=html
}

function start_api() {
    echo "Starting API server..."
    python -m app.main
}

function start_ui() {
    echo "Starting Streamlit UI..."
    streamlit run app/ui/streamlit_app.py
}

function start_ollama() {
    echo "Starting Ollama..."
    ollama serve &
    sleep 3
    echo "Pulling models..."
    ollama pull llama3.1:8b
    ollama pull nomic-embed-text
    echo "Models ready!"
    wait
}

function docker_up() {
    echo "Starting docker stack..."
    cd docker
    docker-compose up -d
    echo "Waiting for services..."
    sleep 5
    docker-compose logs -f model-puller
}

function docker_down() {
    echo "Stopping docker stack..."
    cd docker
    docker-compose down
}

function docker_logs() {
    cd docker
    docker-compose logs -f
}

function clean() {
    echo "Cleaning..."
    rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov dist build *.egg-info
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
}

case "${1:-help}" in
    install) install_deps ;;
    lint) run_lint ;;
    format) run_format ;;
    test) run_tests ;;
    api) start_api ;;
    ui) start_ui ;;
    ollama) start_ollama ;;
    docker-up) docker_up ;;
    docker-down) docker_down ;;
    docker-logs) docker_logs ;;
    clean) clean ;;
    help|*) show_help ;;
esac
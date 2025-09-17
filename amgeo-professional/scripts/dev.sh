#!/bin/bash
# Development environment runner

usage() {
  echo "Usage: $0 {start|test|lint|docker|clean|install}"
  echo ""
  echo "Commands:"
  echo "  start   - Start Streamlit development server"
  echo "  test    - Run test suite"
  echo "  lint    - Run code formatting and linting"
  echo "  docker  - Start full Docker development stack"
  echo "  clean   - Clean up temporary files"
  echo "  install - Install all dependencies"
}

case "$1" in
  "start")
    echo "Starting development environment..."
    docker-compose up -d postgres redis
    poetry run python -m streamlit run streamlit_app/app.py
    ;;
  "test")
    echo "Running tests..."
    poetry run pytest tests/ -v --cov=src/amgeo
    ;;
  "lint")
    echo "Running code quality checks..."
    poetry run black src/ tests/
    poetry run flake8 src/ tests/
    poetry run mypy src/
    ;;
  "docker")
    echo "Starting full Docker development environment..."
    docker-compose up --build
    ;;
  "clean")
    echo "Cleaning up development environment..."
    docker-compose down -v
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    ;;
  "install")
    echo "Installing dependencies..."
    poetry install
    pre-commit install
    ./scripts/setup/install_dependencies.sh
    ;;
  *)
    usage
    ;;
esac

chmod +x scripts/dev.sh

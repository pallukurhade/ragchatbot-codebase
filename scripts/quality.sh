#!/bin/bash
set -e

PYTHON_DIRS="backend/ main.py"

case "${1:-check}" in
  format)
    echo "==> Formatting with black..."
    uv run black $PYTHON_DIRS
    ;;
  check)
    echo "==> Checking formatting (black --check)..."
    uv run black --check $PYTHON_DIRS
    echo "==> Running tests..."
    cd backend && uv run pytest tests/ -v
    ;;
  test)
    echo "==> Running tests..."
    cd backend && uv run pytest tests/ -v
    ;;
  *)
    echo "Usage: $0 [format|check|test]"
    echo "  format  - auto-format all Python files"
    echo "  check   - verify formatting + run tests (default)"
    echo "  test    - run tests only"
    exit 1
    ;;
esac

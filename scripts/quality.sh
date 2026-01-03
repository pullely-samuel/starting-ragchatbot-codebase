#!/bin/bash

# Development quality checks script
# Usage: ./scripts/quality.sh [command]
# Commands:
#   check   - Run all quality checks (lint + format check)
#   fix     - Auto-fix issues (format + lint with fixes)
#   format  - Format code only
#   lint    - Lint code only

set -e

cd "$(dirname "$0")/.."

case "${1:-check}" in
    check)
        echo "Running format check..."
        uv run ruff format --check .
        echo ""
        echo "Running linter..."
        uv run ruff check .
        echo ""
        echo "All quality checks passed!"
        ;;
    fix)
        echo "Formatting code..."
        uv run ruff format .
        echo ""
        echo "Running linter with auto-fix..."
        uv run ruff check --fix .
        echo ""
        echo "Quality fixes applied!"
        ;;
    format)
        echo "Formatting code..."
        uv run ruff format .
        echo "Formatting complete!"
        ;;
    lint)
        echo "Running linter..."
        uv run ruff check .
        echo "Linting complete!"
        ;;
    *)
        echo "Usage: $0 {check|fix|format|lint}"
        echo ""
        echo "Commands:"
        echo "  check   - Run all quality checks (lint + format check)"
        echo "  fix     - Auto-fix issues (format + lint with fixes)"
        echo "  format  - Format code only"
        echo "  lint    - Lint code only"
        exit 1
        ;;
esac

# Variables
VENV_DIR = .venv
UV = uv

# Find all Python files in the project
PY_FILES := src/ft_mlp/*.py

# Colors for output
GREEN = \033[0;32m
YELLOW = \033[0;33m
RED = \033[0;31m
NC = \033[0m # No Color

.PHONY: all norminette test clean fclean venv help

all: $(VENV_DIR)
	@echo -e "$(GREEN)[INFO]$(NC) Setting up virtual environment and dependencies..."
	@echo -e "$(GREEN)[SUCCESS]$(NC) Project setup complete!"
	@echo -e "$(GREEN)[INFO]$(NC) To use the virtual environment, run:"
	@echo -e "$(YELLOW)source .venv/bin/activate$(NC)"

$(VENV_DIR): pyproject.toml uv.lock
	@echo -e "$(YELLOW)[INFO]$(NC) Syncing virtual environment and dependencies..."
	$(UV) sync
	@touch $(VENV_DIR)
	@echo -e "$(GREEN)[SUCCESS]$(NC) Virtual environment created and dependencies installed!"

norminette: $(VENV_DIR)
	@echo -e "$(YELLOW)[INFO]$(NC) Running flake8 (norminette) on all Python files..."
	@if [ -z "$(PY_FILES)" ]; then \
		echo -e "$(YELLOW)[WARNING]$(NC) No Python files found!"; \
	else \
		$(UV) run flake8 $(PY_FILES) && echo -e "$(GREEN)[SUCCESS]$(NC) All files pass norminette!" || echo -e "$(RED)[ERROR]$(NC) Norminette violations found!"; \
	fi

test: $(VENV_DIR)
	@$(UV) run python scripts/run_tests.py

clean:
	@echo -e "$(YELLOW)[INFO]$(NC) Removing Python caches..."
	@find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	@echo -e "$(GREEN)[SUCCESS]$(NC) Caches removed!"

fclean: clean
	@echo -e "$(YELLOW)[INFO]$(NC) Removing virtual environment..."
	@if [ -d "$(VENV_DIR)" ]; then \
		rm -rf $(VENV_DIR); \
		echo -e "$(GREEN)[SUCCESS]$(NC) Virtual environment removed!"; \
	else \
		echo -e "$(YELLOW)[WARNING]$(NC) No virtual environment to remove!"; \
	fi

venv: $(VENV_DIR)
	@echo -e "$(GREEN)[INFO]$(NC) To use the virtual environment, run:"
	@echo -e "$(YELLOW)source .venv/bin/activate$(NC)"

help:
	@echo "Available targets:"
	@echo "  all        - Sync .venv and install dependencies (uv sync)"
	@echo "  venv       - Display how to activate virtual environment"
	@echo "  norminette - Run flake8 on all Python files"
	@echo "  test       - Run all unit tests in tests/ directory"
	@echo "  clean      - Remove Python caches"
	@echo "  fclean     - Remove caches and .venv directory"

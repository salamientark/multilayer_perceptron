# Variables
VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
FLAKE8 = $(VENV_DIR)/bin/flake8

# Find all Python files in the project
PY_FILES := ft_mlp/*.py \
	analyse_data.py \
	split_dataset.py \
	train.py \
	predict.py
			

# Colors for output
GREEN = \033[0;32m
YELLOW = \033[0;33m
RED = \033[0;31m
NC = \033[0m # No Color

.PHONY: all norminette test clean fclean

all: $(VENV_DIR)
	@echo -e "$(GREEN)[INFO]$(NC) Setting up virtual environment and dependencies..."
	@echo -e "$(GREEN)[SUCCESS]$(NC) Project setup complete!"
	@echo -e "$(GREEN)[INFO]$(NC) To use the virtual environment, run:"
	@echo -e "$(YELLOW)source .venv/bin/activate$(NC)"

$(VENV_DIR):
	@echo -e "$(YELLOW)[INFO]$(NC) Creating virtual environment..."
	python3 -m venv $(VENV_DIR)
	@echo -e "$(YELLOW)[INFO]$(NC) Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install numpy matplotlib pandas flake8 PyQt5 seaborn
	@echo -e "$(GREEN)[SUCCESS]$(NC) Virtual environment created and dependencies installed!"

norminette: $(VENV_DIR)
	@echo -e "$(YELLOW)[INFO]$(NC) Running flake8 (norminette) on all Python files..."
	@if [ -z "$(PY_FILES)" ]; then \
		echo -e "$(YELLOW)[WARNING]$(NC) No Python files found!"; \
	else \
		$(FLAKE8) $(PY_FILES) && echo -e "$(GREEN)[SUCCESS]$(NC) All files pass norminette!" || echo -e "$(RED)[ERROR]$(NC) Norminette violations found!"; \
	fi

test: $(VENV_DIR)
	@$(PYTHON) scripts/run_tests.py $(PYTHON)

clean:
	@echo -e "$(YELLOW)[INFO]$(NC) Removing installed dependencies..."
	@if [ -d "$(VENV_DIR)" ]; then \
		$(PIP) freeze | xargs $(PIP) uninstall -y 2>/dev/null || true; \
		echo -e "$(GREEN)[SUCCESS]$(NC) Dependencies removed!"; \
	else \
		echo -e "$(YELLOW)[WARNING]$(NC) No virtual environment found!"; \
	fi

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
	@echo "  all        - Create .venv and install dependencies"
	@echo "  venv       - Display how to activate virtual environment"
	@echo "  norminette - Run flake8 on all Python files"
	@echo "  test       - Run all unit tests in tests/ directory"
	@echo "  clean      - Remove installed dependencies"
	@echo "  fclean     - Remove dependencies and .venv directory"

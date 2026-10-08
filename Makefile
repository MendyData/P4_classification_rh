# =============================================================================
# Makefile - P4 Classification RH
# =============================================================================
# Commandes pour gerer l'environnement et le projet, via uv
# Usage: make <commande>
# =============================================================================

.PHONY: help setup install install-dev install-all clean test test-cov lint format check jupyter update export init

# Variables
PYTHON_VERSION = 3.11
# Nom affiche par le terminal et par VS Code pour le .venv du projet
NOM_ENVIRONNEMENT = P4_AttritionRH

# Sous Windows, make utilise cmd.exe par defaut : les recettes (echo, find, rm)
# sont ecrites pour un shell POSIX, on force donc le sh fourni avec Git.
ifeq ($(OS),Windows_NT)
SHELL := C:/Program Files/Git/bin/sh.exe
endif

# =============================================================================
# AIDE
# =============================================================================

help:
	@echo "=============================================="
	@echo "  P4 Classification RH - Commandes disponibles"
	@echo "=============================================="
	@echo ""
	@echo "INSTALLATION:"
	@echo "  make setup        - Installer Python $(PYTHON_VERSION) avec uv et creer l'environnement"
	@echo "  make install      - Installer les dependances principales"
	@echo "  make install-dev  - Installer avec les deps de developpement"
	@echo "  make install-all  - Installer toutes les dependances"
	@echo ""
	@echo "DEVELOPPEMENT:"
	@echo "  make test         - Lancer les tests"
	@echo "  make lint         - Verifier la qualite du code"
	@echo "  make format       - Formater le code (black + isort)"
	@echo "  make check        - Lancer lint + tests"
	@echo "  make jupyter      - Lancer JupyterLab"
	@echo ""
	@echo "MAINTENANCE:"
	@echo "  make clean        - Nettoyer les fichiers temporaires"
	@echo "  make update       - Mettre a jour les dependances"
	@echo "  make export       - Exporter les dependances vers requirements.txt"
	@echo ""

# =============================================================================
# INSTALLATION
# =============================================================================

# Version de Python lue ensuite par uv dans .python-version
setup:
	@echo "Installation de Python $(PYTHON_VERSION) et creation de l'environnement..."
	uv python install $(PYTHON_VERSION)
	uv venv --allow-existing --prompt $(NOM_ENVIRONNEMENT)
	uv sync
	@echo "Environnement $(NOM_ENVIRONNEMENT) cree dans .venv"

# Dependances de [project] uniquement, sans le groupe dev
install:
	@echo "Installation des dependances principales..."
	uv sync --no-dev
	@echo "Installation terminee!"

# Le groupe dev est installe par defaut par uv sync
install-dev:
	@echo "Installation avec dependances de developpement..."
	uv sync
	@echo "Installation terminee!"

install-all:
	@echo "Installation de toutes les dependances..."
	uv sync --all-groups
	@echo "Installation terminee!"

# =============================================================================
# DEVELOPPEMENT
# =============================================================================

test:
	@echo "Lancement des tests..."
	uv run pytest

test-cov:
	@echo "Lancement des tests avec couverture..."
	uv run pytest --cov=src --cov-report=html

lint:
	@echo "Verification du code..."
	uv run flake8 src tests
	uv run mypy src

format:
	@echo "Formatage du code..."
	uv run black src tests notebooks
	uv run isort src tests

check: lint test
	@echo "Verification complete!"

jupyter:
	@echo "Lancement de JupyterLab..."
	uv run jupyter lab

# =============================================================================
# MAINTENANCE
# =============================================================================

clean:
	@echo "Nettoyage des fichiers temporaires..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true
	@echo "Nettoyage termine!"

# Recalcule uv.lock avec les dernieres versions autorisees, puis synchronise .venv
update:
	@echo "Mise a jour des dependances..."
	uv lock --upgrade
	uv sync

export:
	@echo "Export des dependances vers requirements.txt..."
	uv export --format requirements-txt --no-hashes --output-file requirements.txt
	@echo "Export termine!"

# =============================================================================
# RACCOURCIS
# =============================================================================

# Raccourci pour setup complet
init: setup
	@echo "Environnement pret : uv run jupyter lab"

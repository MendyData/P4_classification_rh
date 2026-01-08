# P4 Classification Rh

Projet de classification RH utilisant le Machine Learning

## Structure du Projet

```
p4_classification_rh/
├── configs/              # Fichiers de configuration
├── data/
│   ├── raw/              # Données brutes
│   ├── processed/        # Données traitées
│   ├── interim/          # Données intermédiaires
│   └── external/         # Données externes
├── docs/                 # Documentation
├── logs/                 # Fichiers de logs
├── models/
│   ├── trained/          # Modèles entraînés
│   └── checkpoints/      # Points de sauvegarde
├── notebooks/
│   ├── exploration/      # Exploration des données
│   ├── modeling/         # Modélisation
│   └── evaluation/       # Évaluation
├── reports/
│   ├── figures/          # Visualisations
│   └── metrics/          # Métriques
├── scripts/              # Scripts utilitaires
├── src/
│   ├── data/             # Chargement des données
│   ├── features/         # Création de features
│   ├── models/           # Modèles ML
│   ├── pipelines/        # Pipelines
│   ├── utils/            # Utilitaires
│   └── visualization/    # Visualisations
└── tests/
    ├── unit/             # Tests unitaires
    └── integration/      # Tests d'intégration
```

##  Installation

### Prérequis
- Python >=3.11
- Poetry (gestionnaire de dépendances)

### Installation de Poetry
```bash
# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Linux/macOS
curl -sSL https://install.python-poetry.org | python3 -
```

### Installation des dépendances
```bash
# Cloner le projet
cd p4_classification_rh

# Installer les dépendances
poetry install

# Activer l'environnement virtuel
poetry shell
```

##  Utilisation

### Activer l'environnement
```bash
poetry shell
```

### Lancer Jupyter Lab
```bash
poetry run jupyter lab
```

### Exécuter les tests
```bash
poetry run pytest
```

### Formater le code
```bash
poetry run black src/
poetry run isort src/
```

## Workflow du Projet

1. **Exploration des données** : `notebooks/exploration/`
2. **Prétraitement** : `src/data/` et `src/features/`
3. **Modélisation** : `notebooks/modeling/` et `src/models/`
4. **Évaluation** : `notebooks/evaluation/`
5. **Production** : `src/pipelines/`

## 👤 Auteur

- **MendyData** - [mendy.data@gmail.com](mailto:mendy.data@gmail.com)

## License

Ce projet est sous licence MIT.

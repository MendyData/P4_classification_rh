# Projet 4 - Prédiction de l'attrition RH chez TechNova Partners

Identification des facteurs de départ des collaborateurs, modèle de classification de la probabilité de démission et interprétation des résultats avec SHAP.

Formation AI Engineer, OpenClassrooms. Auteure : Angèle GOMIS.

---

## 1. Structure du projet

```text
P4_Classification_rh/
├── data/
│   ├── raw/                  extrait_sirh.csv, extrait_eval.csv, extrait_sondage.csv
│   ├── interim/              consolide.parquet (notebook 01)
│   └── processed/            X.parquet, y.parquet (notebook 01), split_indices.json (notebook 02)
├── models/                   modele_final.joblib, metadonnees_modele.json (notebook 02)
├── notebooks/
│   ├── 01_exploration_features_final_0210.ipynb  étapes 1 et 2 : exploration, 3 features créées, X et y
│   ├── 02_modelisation_final_0210.ipynb       étapes 3, 4 et 5 : comparaison de tous les modèles, modèle final
│   └── 03_interpretation_final_0210.ipynb     étape 5 : importance globale et locale
├── reports/                  dictionnaire_variables.csv, facteurs_actions.csv, fiches/ (application métier)
├── src/
│   ├── __init__.py
│   ├── application_metier.py fiches collaborateur et liste des alertes pour les RH
│   └── config.py             chemins et constantes partagés par les notebooks et l'application
├── .python-version           version de Python utilisée par uv
├── pyproject.toml            dépendances et configuration des outils
├── uv.lock                   versions exactes installées, générées par uv
└── README.md
```

Les notebooks s'exécutent dans l'ordre 01, 02, 03 : chacun relit les fichiers produits par le précédent.

| Notebook | Contenu | Découpage |
|---|---|---|
| 01 | Chargement, nettoyage, jointure, statistiques descriptives, graphiques, encodage, corrélations, construction de `X` et `y` | Aucun apprentissage |
| 02 | Modèles de référence, pondération des classes, features supplémentaires, SMOTE, seuil de décision, recherche d'hyperparamètres, confirmation sur le test | Entraînement 60 %, validation 20 %, test 20 %, stratifiés |
| 03 | Importance native, permutation, SHAP (beeswarm, scatter, waterfall), synthèse des leviers RH | Jeu de test |

---

## 2. Installation avec uv

uv remplace Poetry pour gérer Python, l'environnement virtuel et les dépendances. Toutes les commandes se lancent depuis la racine du projet, dans l'invite de commandes Windows ou PowerShell.

### 2.1 Installer uv (une seule fois sur le poste)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Fermer puis rouvrir le terminal, et vérifier l'installation :

```bat
uv --version
```

### 2.2 Installer Python 3.11 (une seule fois sur le poste)

```bat
uv python install 3.11
```

uv lit ensuite le fichier `.python-version` et utilise automatiquement cette version pour le projet.

### 2.3 Créer l'environnement du projet

```bat
cd /d "C:\Users\Windows 11\Desktop\AI_Engineer\P4_Classification_rh"
uv venv --prompt P4_AttritionRH
uv sync
```

`uv venv --prompt P4_AttritionRH` crée le dossier `.venv` sous le nom `P4_AttritionRH`, affiché dans le terminal et dans le sélecteur d'interpréteur de VS Code. `uv sync` installe les dépendances du projet et du groupe `dev` (Jupyter, black, mypy, pytest), installe le dossier `src` comme paquet et écrit `uv.lock` s'il n'existe pas encore.

### 2.4 Vérifier l'installation

```bat
uv run python -c "from src import config; print(config.RACINE_PROJET)"
```

La commande doit afficher le chemin du projet. Si elle échoue sur `src`, vérifier que le fichier `src/__init__.py` existe, même vide.

---

## 3. Exécution des notebooks

Interface Jupyter :

```bat
uv run jupyter lab
```

Exécution complète en ligne de commande, les sorties étant réécrites dans chaque fichier :

```bat
uv run jupyter nbconvert --to notebook --execute --inplace notebooks\01_exploration_features_final_0210.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks\02_modelisation_final_0210.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks\03_interpretation_final_0210.ipynb
```

Si le noyau du projet n'apparaît pas dans la liste de Jupyter ou de VS Code, l'enregistrer une fois :

```bat
uv run python -m ipykernel install --user --name p4_attritionrh --display-name "P4_AttritionRH"
```

---

## 4. Qualité du code

```bat
uv run black notebooks src
uv run mypy src
uv run pytest
```

---

## 5. Résultats principaux

- Jointure interne un-à-un sur `id_employee` : 1 470 collaborateurs, 16,1 % de départs.
- Trois features créées à partir des données existantes (`FEATURES_CREEES` dans `src/config.py`) : `satisfaction_minimale`, `nouveau_responsable` (32,3 % de départs la première année sous un nouveau responsable, contre 12,6 %) et `debut_de_carriere` (43,9 % de départs avec 2 ans d'expérience ou moins, contre 13,6 %).
- Comparaison de tous les modèles (régression logistique par défaut et réglée, forêts aléatoires, XGBoost pondéré et réglé), réentraînés sur l'entraînement et la validation réunis, avec un seuil fixé sur des prédictions hors pli.
- Règle de choix fixée avec le métier (`RAPPEL_MINIMAL` dans `src/config.py`) : détecter au moins 80 % des départs, et à ce niveau émettre le moins de fausses alertes.
- Modèle retenu : régression logistique pondérée, seuil de décision 0,418.
- Sur le jeu de test : 36 départs détectés sur 47 (rappel de 0,766) pour 63 fausses alertes ; un peu plus d'une alerte sur trois correspond à un départ réel.
- Principaux facteurs de départ : heures supplémentaires, satisfaction au plus bas sur au moins une dimension, premières années (début de carrière, changement récent de responsable), promotion ancienne, trajets et déplacements ; risque nettement plus élevé sous 3 000 € de salaire mensuel.

---

## 6. Application métier

L'application part de l'alerte du modèle et montre aux RH le profil du collaborateur en clair, sur une page : le profil à gauche, et à droite les facteurs qui pèsent le plus sur son score, en **vert** s'ils correspondent à un levier actionnable par les RH, en **rouge** sinon, avec les leviers à aborder en entretien. Elle utilise le modèle et les données produits par les notebooks : exécuter d'abord les notebooks 01 et 02.

Fiche d'un ou plusieurs collaborateurs, à partir de leur identifiant (`id_employee`) :

```bat
uv run python -m src.application_metier 1720 331
```

Liste des collaborateurs présents en alerte, du plus au moins exposé, avec le premier levier à aborder (`reports/fiches/alertes.csv`, au format Excel français), et fiches des 20 plus exposés :

```bat
uv run python -m src.application_metier --alertes --max-fiches 20
```

Précautions intégrées à l'application :
- le genre, le statut marital et l'âge sont pris en compte par le modèle mais n'apparaissent jamais sur une fiche ;
- le score est présenté comme un score à comparer au seuil d'alerte, pas comme une probabilité de départ ;
- la fiche signale les collaborateurs ayant servi à l'entraînement, dont le score est optimiste ;
- les fiches contiennent des données personnelles : `reports/fiches/` est exclu du dépôt git.

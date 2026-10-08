"""
Constantes uniques du projet : chemins, cible, colonnes, couleurs, coûts.

Toute constante utilisée dans plus d'un notebook ou module est définie ici et
importée, jamais redéfinie ailleurs. Les noms de colonnes ci-dessous ont été
confrontés aux en-têtes réels des trois extraits sources avant leur écriture.
"""

from __future__ import annotations

from pathlib import Path

# =============================================================================
# Arborescence
# =============================================================================

RACINE_PROJET: Path = Path(__file__).resolve().parent.parent

DOSSIER_DONNEES_BRUTES: Path = RACINE_PROJET / "data" / "raw"
DOSSIER_DONNEES_INTERMEDIAIRES: Path = RACINE_PROJET / "data" / "interim"
DOSSIER_DONNEES_TRAITEES: Path = RACINE_PROJET / "data" / "processed"
# --- Fichiers produits par le notebook 01 (etape 2) et le notebook 02 (etape 4) ---
# Features pretes pour fit(), apres encodage et retrait des variables redondantes
FICHIER_X = DOSSIER_DONNEES_TRAITEES / "X.parquet"
# Cible binaire : 1 = depart, 0 = reste
FICHIER_Y = DOSSIER_DONNEES_TRAITEES / "y.parquet"
# Features de X completees par les trois features metier de l'etape 4
FICHIER_X_ENRICHI = DOSSIER_DONNEES_TRAITEES / "X_enrichi.parquet"

DOSSIER_MODELES: Path = RACINE_PROJET / "models"
DOSSIER_RAPPORTS: Path = RACINE_PROJET / "reports"

FICHIER_SIRH: Path = DOSSIER_DONNEES_BRUTES / "extrait_sirh.csv"
FICHIER_EVAL: Path = DOSSIER_DONNEES_BRUTES / "extrait_eval.csv"
FICHIER_SONDAGE: Path = DOSSIER_DONNEES_BRUTES / "extrait_sondage.csv"

# Les trois extraits sont encodés en UTF-8 (vérifié sur les octets bruts des
# fichiers : \xc3\xa9 = encodage UTF-8 de "é"). Explicité ici plutôt que
# laissé au défaut pandas pour ne pas dépendre d'une détection implicite.
ENCODAGE_SOURCE: str = "utf-8"

# Artefacts échangés entre notebooks (section 3 du cadrage) : seule interface
# autorisée entre eux, aucune variable en mémoire n'est partagée.
FICHIER_CONSOLIDE: Path = DOSSIER_DONNEES_INTERMEDIAIRES / "consolide.parquet"
FICHIER_DICTIONNAIRE_VARIABLES: Path = DOSSIER_RAPPORTS / "dictionnaire_variables.csv"
FICHIER_TESTS_BIVARIES: Path = DOSSIER_RAPPORTS / "tests_bivaries.csv"
FICHIER_SPLIT_INDICES: Path = DOSSIER_DONNEES_TRAITEES / "split_indices.json"
FICHIER_MODELE_FINAL: Path = DOSSIER_MODELES / "modele_final.joblib"
FICHIER_METADONNEES_MODELE: Path = DOSSIER_MODELES / "metadonnees_modele.json"
FICHIER_FACTEURS_ACTIONS: Path = DOSSIER_RAPPORTS / "facteurs_actions.csv"

RANDOM_STATE: int = 42

# =============================================================================
# Cible
# =============================================================================

CIBLE: str = "a_quitte_l_entreprise"
MODALITE_DEPART: str = "Oui"
MODALITE_RESTE: str = "Non"

# =============================================================================
# Clés de jointure
# =============================================================================

CLE_SIRH: str = "id_employee"
CLE_EVAL: str = "eval_number"
CLE_SONDAGE: str = "code_sondage"
COLONNES_IDENTIFIANTS: tuple[str, ...] = (CLE_SIRH, CLE_EVAL, CLE_SONDAGE)

# =============================================================================
# Colonnes à formats bruts, à corriger avec .apply() avant toute jointure
# =============================================================================

# Format réel "E_<entier>" (et non "eval_XXX" comme envisagé initialement).
COLONNE_EVAL_NUMBER_BRUTE: str = "eval_number"
# Format réel "<entier> %" avec espace avant le signe pourcentage.
COLONNE_AUGMENTATION_SALAIRE_BRUTE: str = "augementation_salaire_precedente"

# =============================================================================
# Colonnes sans pouvoir informatif constatées sur les trois extraits
# =============================================================================

# nombre_heures_travailless vaut 80, ayant_enfants vaut "Y" et
# nombre_employee_sous_responsabilite vaut la même valeur pour les 1470
# lignes des extraits concernés. Constatées au chargement (notebook 01),
# retirées au motif qu'une variable sans variance ne peut discriminer aucun
# comportement.
COLONNES_SANS_VARIANCE: tuple[str, ...] = (
    "nombre_heures_travailless",
    "ayant_enfants",
    "nombre_employee_sous_responsabilite",
)

# =============================================================================
# Colonnes de profil (âge, expérience, revenu)
# =============================================================================

COL_AGE: str = "age"
COL_EXPERIENCE_TOTALE: str = "annee_experience_totale"
COL_REVENU_MENSUEL: str = "revenu_mensuel"

# =============================================================================
# Colonnes d'ancienneté
# =============================================================================

COL_ANCIENNETE_ENTREPRISE: str = "annees_dans_l_entreprise"
COL_ANCIENNETE_POSTE: str = "annees_dans_le_poste_actuel"
COL_DEPUIS_PROMOTION: str = "annees_depuis_la_derniere_promotion"
COL_SOUS_RESPONSABLE_ACTUEL: str = "annes_sous_responsable_actuel"

# =============================================================================
# Features créées à partir des données existantes (notebook 01, point 12)
# =============================================================================

# Minimum des quatre notes de satisfaction : le point de rupture du collaborateur.
COL_SATISFACTION_MINIMALE: str = "satisfaction_minimale"
# 1 si le collaborateur a changé de responsable depuis moins d'un an
# (COL_SOUS_RESPONSABLE_ACTUEL vaut 0), 0 sinon.
COL_NOUVEAU_RESPONSABLE: str = "nouveau_responsable"
# 1 si l'expérience professionnelle totale ne dépasse pas SEUIL_DEBUT_DE_CARRIERE
# années, 0 sinon. Critère professionnel, volontairement indépendant de l'âge.
COL_DEBUT_DE_CARRIERE: str = "debut_de_carriere"
# Seuil fixé par convention métier (« début de carrière », « jeune diplômé »),
# et non optimisé sur les données, pour ne pas rendre le gain mesuré optimiste.
SEUIL_DEBUT_DE_CARRIERE: int = 2
FEATURES_CREEES: tuple[str, ...] = (
    COL_SATISFACTION_MINIMALE,
    COL_NOUVEAU_RESPONSABLE,
    COL_DEBUT_DE_CARRIERE,
)

# =============================================================================
# Variables potentiellement discriminantes, à faire arbitrer par le commanditaire
# =============================================================================

COLONNES_SENSIBLES: tuple[str, ...] = ("genre", "statut_marital", "age")

# =============================================================================
# Leviers RH associés aux variables (notebook 03 et fiche collaborateur)
# =============================================================================

# Pour chaque variable d'origine : (levier RH, actionnable par les RH). Une
# variable absente de ce dictionnaire délimite la population à suivre sans être
# un levier : elle reçoit LEVIER_PAR_DEFAUT.
LEVIERS_RH: dict[str, tuple[str, bool]] = {
    "heure_supplementaires": (
        "Charge de travail : limiter et repartir les heures supplementaires",
        True,
    ),
    "revenu_mensuel": ("Remuneration : revoir les salaires des bas de grille", True),
    COL_SATISFACTION_MINIMALE: (
        "Suivi de la satisfaction : traiter le point de rupture de chaque collaborateur",
        True,
    ),
    "satisfaction_employee_environnement": ("Environnement de travail", True),
    "satisfaction_employee_nature_travail": ("Contenu des missions", True),
    "satisfaction_employee_equilibre_pro_perso": (
        "Equilibre vie professionnelle / vie personnelle",
        True,
    ),
    "satisfaction_employee_equipe": ("Relations d'equipe et management", True),
    "frequence_deplacement": (
        "Organisation du travail : limiter les deplacements frequents",
        True,
    ),
    "poste": ("Parcours par metier : cibler les postes les plus exposes", True),
    "nombre_participation_pee": (
        "Indicateur d'engagement a suivre, pas un levier direct",
        False,
    ),
    COL_DEPUIS_PROMOTION: ("Evolution de carriere et promotions", True),
    COL_ANCIENNETE_POSTE: ("Mobilite interne", True),
    COL_SOUS_RESPONSABLE_ACTUEL: ("Stabilite du management", True),
    COL_NOUVEAU_RESPONSABLE: (
        "Stabilite du management : accompagner chaque changement de responsable",
        True,
    ),
    COL_DEBUT_DE_CARRIERE: (
        "Integration et fidelisation des debutants : parcours des deux premieres annees",
        True,
    ),
    "nb_formations_suivies": ("Formation", True),
    "distance_domicile_travail": ("Teletravail pour les trajets longs", True),
    "augementation_salaire_precedente": ("Politique de revalorisation salariale", True),
}
LEVIER_PAR_DEFAUT: tuple[str, bool] = ("Non actionnable : delimite la population a suivre", False)

# =============================================================================
# Couleurs communes aux trois notebooks
# =============================================================================

# Bleu pour les collaborateurs restés, rouge pour les partants : code couleur
# tenu identique sur tous les graphiques du projet pour une lecture immédiate.
COULEUR_RESTE: str = "#2563EB"
COULEUR_DEPART: str = "#DC2626"
PALETTE_CIBLE: dict[str, str] = {MODALITE_RESTE: COULEUR_RESTE, MODALITE_DEPART: COULEUR_DEPART}

# =============================================================================
# Hypothèses de coût métier (seuil de décision)
# =============================================================================

# Coûts non communiqués par le commanditaire à ce stade. Le seuil de décision
# est donc optimisé par le F2 (BETA_F2) sur le jeu de validation. Dès que le
# coût d'un entretien de rétention et le coût d'un remplacement seront
# chiffrés, ces deux constantes seront renseignées et le seuil recalculé sur
# le coût métier réel plutôt que sur le F2.
COUT_ENTRETIEN_RETENTION: float | None = None
COUT_REMPLACEMENT: float | None = None
BETA_F2: float = 2.0

# Préférence du commanditaire : une fausse alerte coûte moins cher qu'un départ
# manqué. Le modèle final doit détecter au moins 80 % des départs (rappel hors
# pli) ; à ce niveau, on retient le modèle et le seuil qui émettent le moins de
# fausses alertes.
RAPPEL_MINIMAL: float = 0.80

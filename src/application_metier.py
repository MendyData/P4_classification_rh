"""
Application métier RH : à partir de l'alerte du modèle, le profil du collaborateur en clair.

Pour chaque collaborateur, l'application produit une fiche d'une page : le profil
à gauche, et à droite les facteurs qui pèsent le plus sur son score, colorés selon
leur levier RH (config.LEVIERS_RH) : vert s'il est actionnable, rouge sinon. Elle
peut aussi lister les collaborateurs présents en alerte, du plus au moins exposé.

La fiche reprend le modèle final sauvegardé par le notebook 02 et l'explique avec
les valeurs de Shapley, comme le notebook 03. Les variables sensibles
(config.COLONNES_SENSIBLES) sont prises en compte par le modèle mais n'apparaissent
jamais, ni dans le profil ni dans les facteurs : une fiche remise aux RH ne doit pas
pouvoir motiver une décision fondée sur le genre, le statut marital ou l'âge.

Usage, depuis la racine du projet :
    uv run python -m src.application_metier 1720 331
    uv run python -m src.application_metier --alertes --max-fiches 20
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src import config

DOSSIER_FICHES: Path = config.DOSSIER_RAPPORTS / "fiches"
FICHIER_ALERTES: Path = DOSSIER_FICHES / "alertes.csv"
COULEUR_ACTIONNABLE: str = "#16A34A"
# Rouge plus sombre que COULEUR_DEPART : ici le rouge signifie « non actionnable »,
# pas « parti », et les deux codes couleur ne doivent pas se confondre.
COULEUR_NON_ACTIONNABLE: str = "#B91C1C"
NB_FACTEURS_AFFICHES: int = 10
NB_LEVIERS_PROPOSES: int = 4
NB_FICHES_ALERTES: int = 20

# Libellés courts des variables affichées, dans l'ordre de lecture du profil
SECTIONS_PROFIL: dict[str, dict[str, str]] = {
    "Poste": {
        "poste": "Poste",
        "departement": "Departement",
        "annees_dans_le_poste_actuel": "Annees dans le poste",
        config.COL_SOUS_RESPONSABLE_ACTUEL: "Annees sous le responsable actuel",
    },
    "Parcours": {
        "annees_dans_l_entreprise": "Anciennete dans l'entreprise",
        "annee_experience_totale": "Experience totale",
        "nombre_experiences_precedentes": "Employeurs precedents",
        "annees_depuis_la_derniere_promotion": "Annees depuis la derniere promotion",
        "nb_formations_suivies": "Formations suivies",
        "domaine_etude": "Domaine d'etudes",
    },
    "Remuneration": {
        "revenu_mensuel": "Revenu mensuel",
        "augementation_salaire_precedente": "Derniere augmentation",
        "nombre_participation_pee": "Participations au PEE",
    },
    "Conditions de travail": {
        "heure_supplementaires": "Heures supplementaires",
        "frequence_deplacement": "Deplacements",
        "distance_domicile_travail": "Distance domicile-travail",
    },
    "Satisfaction et evaluation": {
        "satisfaction_employee_environnement": "Satisfaction environnement",
        "satisfaction_employee_nature_travail": "Satisfaction nature du travail",
        "satisfaction_employee_equipe": "Satisfaction equipe",
        "satisfaction_employee_equilibre_pro_perso": "Satisfaction equilibre pro / perso",
        "note_evaluation_precedente": "Note d'evaluation precedente",
        "note_evaluation_actuelle": "Note d'evaluation actuelle",
    },
}
LIBELLES_FEATURES_CREEES: dict[str, str] = {
    config.COL_SATISFACTION_MINIMALE: "Satisfaction minimale",
    config.COL_NOUVEAU_RESPONSABLE: "Nouveau responsable",
    config.COL_DEBUT_DE_CARRIERE: "Debut de carriere",
}
UNITES: dict[str, str] = {
    "annees_dans_le_poste_actuel": " an(s)",
    config.COL_SOUS_RESPONSABLE_ACTUEL: " an(s)",
    "annees_dans_l_entreprise": " an(s)",
    "annee_experience_totale": " an(s)",
    "annees_depuis_la_derniere_promotion": " an(s)",
    "revenu_mensuel": " EUR",
    "augementation_salaire_precedente": " %",
    "distance_domicile_travail": " km",
}
AVERTISSEMENTS: tuple[str, ...] = (
    "Score du modele pondere : a comparer au seuil d'alerte, ce n'est pas une probabilite "
    "de depart.",
    "Genre, statut marital et age sont pris en compte par le modele mais volontairement "
    "non affiches.",
    "Outil d'aide a la discussion : aucune decision individuelle ne doit reposer sur ce "
    "seul score.",
)
AVERTISSEMENT_ENTRAINEMENT: str = (
    "Collaborateur ayant servi a l'entrainement du modele : son score est optimiste."
)


@dataclass(frozen=True, slots=True)
class ContexteModele:
    """Modèle final, données et explicateur SHAP, chargés une seule fois."""

    modele: BaseEstimator
    seuil: float
    X: pd.DataFrame
    df: pd.DataFrame
    index_test: frozenset[int]
    explainer: shap.Explainer
    variables_nominales: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FicheCollaborateur:
    """Tout ce qu'affiche la fiche d'un collaborateur."""

    identifiant: int
    profil: pd.Series
    score: float
    seuil: float
    dans_le_test: bool
    # Une ligne par variable d'origine : contribution, valeur lue, levier, actionnable
    facteurs: pd.DataFrame


def charger_contexte() -> ContexteModele:
    """Recharge les fichiers produits par les notebooks 01 et 02."""
    modele = joblib.load(config.FICHIER_MODELE_FINAL)
    metadonnees = json.loads(config.FICHIER_METADONNEES_MODELE.read_text(encoding="utf-8"))
    separation = json.loads(config.FICHIER_SPLIT_INDICES.read_text(encoding="utf-8"))
    X = pd.read_parquet(config.FICHIER_X)[metadonnees["features"]]
    df = pd.read_parquet(config.FICHIER_CONSOLIDE)

    if isinstance(modele, Pipeline):
        # Modèle linéaire : SHAP explique la régression sur les features mises à l'échelle,
        # avec pour référence les collaborateurs sur lesquels le modèle a été ajusté
        reference = X.loc[separation["index_entrainement"] + separation["index_validation"]]
        explainer = shap.LinearExplainer(
            modele[-1],
            shap.maskers.Independent(modele[:-1].transform(reference), max_samples=len(reference)),
        )
    else:
        explainer = shap.TreeExplainer(modele)

    variables_nominales = tuple(
        colonne
        for colonne in df.columns
        if colonne not in X.columns
        and colonne not in (config.CIBLE, config.CLE_SIRH)
        and not pd.api.types.is_numeric_dtype(df[colonne])
    )
    return ContexteModele(
        modele=modele,
        seuil=metadonnees["seuil_decision"],
        X=X,
        df=df,
        index_test=frozenset(separation["index_test"]),
        explainer=explainer,
        variables_nominales=variables_nominales,
    )


def _contributions(contexte: ContexteModele, X_jeu: pd.DataFrame) -> pd.DataFrame:
    """Contributions SHAP par variable d'origine, sans les variables sensibles."""
    donnees = (
        contexte.modele[:-1].transform(X_jeu) if isinstance(contexte.modele, Pipeline) else X_jeu
    )
    valeurs = pd.DataFrame(
        contexte.explainer(donnees).values, index=X_jeu.index, columns=X_jeu.columns
    )
    # Les colonnes one-hot d'une même variable sont additionnées, comme au notebook 03
    par_variable = valeurs.T.groupby(
        lambda colonne: next(
            (v for v in contexte.variables_nominales if colonne.startswith(f"{v}_")), colonne
        )
    ).sum()
    return par_variable.T.drop(columns=list(config.COLONNES_SENSIBLES), errors="ignore")


def _formater_valeur(variable: str, valeur: object) -> str:
    """Valeur lisible par un RH : unités, oui / non pour les indicateurs créés."""
    if variable in (config.COL_NOUVEAU_RESPONSABLE, config.COL_DEBUT_DE_CARRIERE):
        return "oui" if valeur == 1 else "non"
    if variable == "revenu_mensuel":
        return f"{int(valeur):,}".replace(",", " ") + UNITES[variable]
    if isinstance(valeur, (int, np.integer, float, np.floating)):
        valeur = int(valeur) if float(valeur).is_integer() else round(float(valeur), 2)
    return f"{valeur}{UNITES.get(variable, '')}"


def _decimal(valeur: float) -> str:
    """Nombre à deux décimales, avec la virgule française."""
    return f"{valeur:.2f}".replace(".", ",")


def _index_collaborateur(contexte: ContexteModele, identifiant: int) -> int:
    """Ligne du collaborateur dans les données, à partir de son id_employee."""
    lignes = contexte.df.index[contexte.df[config.CLE_SIRH] == identifiant]
    if lignes.empty:
        raise ValueError(f"Identifiant de collaborateur inconnu : {identifiant}")
    return int(lignes[0])


def expliquer_collaborateur(contexte: ContexteModele, identifiant: int) -> FicheCollaborateur:
    """Score, profil et facteurs de risque d'un collaborateur, identifié par id_employee."""
    index = _index_collaborateur(contexte, identifiant)
    x = contexte.X.loc[[index]]
    contributions = _contributions(contexte, x).iloc[0]

    # Valeurs lisibles du consolidé, complétées des features créées présentes dans X
    profil = pd.concat([contexte.df.loc[index], contexte.X.loc[index]])
    profil = profil[~profil.index.duplicated()]
    leviers = [config.LEVIERS_RH.get(v, config.LEVIER_PAR_DEFAUT) for v in contributions.index]
    facteurs = pd.DataFrame(
        {
            "contribution": contributions,
            "valeur": [_formater_valeur(v, profil[v]) for v in contributions.index],
            "levier": [levier for levier, _ in leviers],
            "actionnable": [actionnable for _, actionnable in leviers],
        }
    ).reindex(contributions.abs().sort_values(ascending=False).index)

    return FicheCollaborateur(
        identifiant=identifiant,
        profil=profil,
        score=float(contexte.modele.predict_proba(x)[0, 1]),
        seuil=contexte.seuil,
        dans_le_test=index in contexte.index_test,
        facteurs=facteurs,
    )


def lister_alertes(contexte: ContexteModele) -> pd.DataFrame:
    """Collaborateurs présents dont le score atteint le seuil, du plus au moins exposé."""
    # Un entretien de rétention n'a de sens que pour un collaborateur encore présent
    presents = contexte.X.loc[contexte.df[config.CIBLE].eq(config.MODALITE_RESTE)]
    scores = pd.Series(contexte.modele.predict_proba(presents)[:, 1], index=presents.index)
    en_alerte = scores[scores >= contexte.seuil].sort_values(ascending=False)
    contributions = _contributions(contexte, contexte.X.loc[en_alerte.index])
    actionnables = contributions[
        [v for v in contributions.columns if config.LEVIERS_RH.get(v, config.LEVIER_PAR_DEFAUT)[1]]
    ]
    # Premier levier : le facteur actionnable qui pousse le plus vers le départ
    premier = actionnables.idxmax(axis=1)
    return pd.DataFrame(
        {
            config.CLE_SIRH: contexte.df.loc[en_alerte.index, config.CLE_SIRH],
            "poste": contexte.df.loc[en_alerte.index, "poste"],
            "departement": contexte.df.loc[en_alerte.index, "departement"],
            "score": en_alerte.round(3),
            "jeu": [
                "test" if i in contexte.index_test else "entrainement" for i in en_alerte.index
            ],
            "premier_levier": [
                config.LEVIERS_RH[variable][0] if actionnables.loc[i, variable] > 0 else ""
                for i, variable in premier.items()
            ],
        }
    ).reset_index(drop=True)


def _libelle(variable: str) -> str:
    """Libellé court d'une variable, à défaut son nom technique."""
    for section in SECTIONS_PROFIL.values():
        if variable in section:
            return section[variable]
    return LIBELLES_FEATURES_CREEES.get(variable, variable)


def _tracer_profil(ax: Axes, fiche: FicheCollaborateur) -> None:
    """Colonne de gauche : niveau de risque, puis profil du collaborateur par sections."""
    ax.axis("off")
    alerte = fiche.score >= fiche.seuil
    ax.text(
        0, 1.0, f"Collaborateur n° {fiche.identifiant}", fontsize=17, fontweight="bold", va="top"
    )
    ax.text(
        0,
        0.94,
        "ALERTE : entretien recommande" if alerte else "Pas d'alerte au seuil retenu",
        fontsize=12,
        fontweight="bold",
        color="white",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": config.COULEUR_DEPART if alerte else "grey",
        },
    )
    ax.text(
        0,
        0.885,
        f"Score de risque {_decimal(fiche.score)}  |  seuil d'alerte {_decimal(fiche.seuil)}",
        fontsize=11,
        va="top",
    )

    y = 0.83
    for titre, variables in SECTIONS_PROFIL.items():
        ax.text(0, y, titre, fontsize=11.5, fontweight="bold", va="top", color="#1F2937")
        y -= 0.032
        for variable, libelle in variables.items():
            valeur = _formater_valeur(variable, fiche.profil[variable])
            ax.text(0.02, y, libelle, fontsize=9.5, va="top", color="#374151")
            ax.text(0.98, y, valeur, fontsize=9.5, va="top", ha="right", fontweight="bold")
            y -= 0.026
        y -= 0.01
    indicateurs = ", ".join(
        f"{LIBELLES_FEATURES_CREEES[v].lower()} : {_formater_valeur(v, fiche.profil[v])}"
        for v in (config.COL_NOUVEAU_RESPONSABLE, config.COL_DEBUT_DE_CARRIERE)
    )
    ax.text(0, y, f"Indicateurs : {indicateurs}", fontsize=9.5, va="top", style="italic")


def _tracer_facteurs(ax: Axes, fiche: FicheCollaborateur) -> None:
    """Haut de la colonne de droite : contributions des principaux facteurs, vert ou rouge."""
    principaux = fiche.facteurs.head(NB_FACTEURS_AFFICHES).iloc[::-1]
    ax.barh(
        [f"{_libelle(v)} ({valeur})" for v, valeur in zip(principaux.index, principaux["valeur"])],
        principaux["contribution"],
        color=[
            COULEUR_ACTIONNABLE if actionnable else COULEUR_NON_ACTIONNABLE
            for actionnable in principaux["actionnable"]
        ],
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(
        "<- retient le collaborateur        Contribution au score (log-odds)"
        "        pousse vers le depart ->"
    )
    ax.set_title(
        f"Les {len(principaux)} facteurs qui pesent le plus pour ce collaborateur", fontsize=12
    )
    ax.tick_params(axis="y", labelsize=9.5)
    ax.legend(
        handles=[
            Patch(color=COULEUR_ACTIONNABLE, label="Levier actionnable par les RH"),
            Patch(color=COULEUR_NON_ACTIONNABLE, label="Non actionnable : delimite la population"),
        ],
        loc="lower right",
        fontsize=9,
    )


def _tracer_leviers(ax: Axes, fiche: FicheCollaborateur) -> None:
    """Bas de la colonne de droite : les leviers à aborder en entretien, puis les avertissements."""
    ax.axis("off")
    a_discuter = fiche.facteurs[
        (fiche.facteurs["contribution"] > 0) & fiche.facteurs["actionnable"]
    ]
    ax.text(0, 1.0, "Leviers a aborder en entretien", fontsize=12, fontweight="bold", va="top")
    if a_discuter.empty:
        ax.text(
            0, 0.85, "Aucun facteur actionnable ne pousse vers le depart.", fontsize=10, va="top"
        )
    for rang, (variable, ligne) in enumerate(a_discuter.head(NB_LEVIERS_PROPOSES).iterrows()):
        ax.text(
            0,
            0.85 - rang * 0.13,
            f"{rang + 1}. {_libelle(variable)} ({ligne['valeur']}) : {ligne['levier']}",
            fontsize=10,
            va="top",
            color=COULEUR_ACTIONNABLE,
        )

    avertissements = list(AVERTISSEMENTS)
    if not fiche.dans_le_test:
        avertissements.append(AVERTISSEMENT_ENTRAINEMENT)
    for rang, texte in enumerate(avertissements):
        ax.text(
            0, 0.28 - rang * 0.09, texte, fontsize=8.5, va="top", color="#6B7280", style="italic"
        )


def tracer_fiche(fiche: FicheCollaborateur) -> Figure:
    """Page complète : profil à gauche, facteurs et leviers à droite."""
    figure = Figure(figsize=(17, 10), dpi=110, layout="constrained")
    grille = figure.add_gridspec(2, 2, width_ratios=[1, 1.7], height_ratios=[2.1, 1])
    _tracer_profil(figure.add_subplot(grille[:, 0]), fiche)
    _tracer_facteurs(figure.add_subplot(grille[0, 1]), fiche)
    _tracer_leviers(figure.add_subplot(grille[1, 1]), fiche)
    return figure


def enregistrer_fiche(
    contexte: ContexteModele, identifiant: int, dossier: Path = DOSSIER_FICHES
) -> Path:
    """Génère la fiche d'un collaborateur et l'enregistre au format PNG."""
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / f"fiche_{identifiant}.png"
    tracer_fiche(expliquer_collaborateur(contexte, identifiant)).savefig(chemin)
    return chemin


def main() -> None:
    """Point d'entrée en ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Application metier RH : fiches collaborateur et alertes du modele."
    )
    parser.add_argument("identifiants", nargs="*", type=int, help="id_employee des collaborateurs")
    parser.add_argument(
        "--alertes",
        action="store_true",
        help="liste les collaborateurs en alerte et genere la fiche des plus exposes",
    )
    parser.add_argument(
        "--max-fiches",
        type=int,
        default=NB_FICHES_ALERTES,
        help=f"nombre de fiches generees avec --alertes (defaut : {NB_FICHES_ALERTES})",
    )
    arguments = parser.parse_args()
    if not arguments.identifiants and not arguments.alertes:
        parser.error("indiquer au moins un identifiant, ou l'option --alertes")

    contexte = charger_contexte()
    identifiants = list(arguments.identifiants)
    if arguments.alertes:
        alertes = lister_alertes(contexte)
        DOSSIER_FICHES.mkdir(parents=True, exist_ok=True)
        # Format Excel francais : separateur point-virgule, virgule decimale, accents conserves
        alertes.to_csv(FICHIER_ALERTES, index=False, sep=";", decimal=",", encoding="utf-8-sig")
        print(f"{len(alertes)} collaborateurs presents en alerte : {FICHIER_ALERTES}")
        identifiants += alertes[config.CLE_SIRH].head(arguments.max_fiches).tolist()
    for identifiant in identifiants:
        try:
            print(f"Fiche : {enregistrer_fiche(contexte, identifiant)}")
        except ValueError as erreur:
            # Un identifiant inconnu ne doit pas empecher de produire les autres fiches
            print(f"Fiche non produite : {erreur}")


if __name__ == "__main__":
    main()

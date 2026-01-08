"""
=============================================================================
Module de prétraitement des données
=============================================================================

Ce module contient les fonctions de nettoyage et prétraitement des données.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from loguru import logger


def clean_data(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    drop_na_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Nettoie les données en supprimant les doublons et colonnes vides.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à nettoyer.
    drop_duplicates : bool
        Si True, supprime les lignes dupliquées.
    drop_na_threshold : float
        Seuil de valeurs manquantes pour supprimer une colonne.
    
    Returns
    -------
    pd.DataFrame
        DataFrame nettoyé.
    """
    df_clean = df.copy()
    initial_shape = df_clean.shape
    
    # Suppression des doublons
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Doublons supprimés: {initial_shape[0] - df_clean.shape[0]}")
    
    # Suppression des colonnes avec trop de valeurs manquantes
    na_ratio = df_clean.isnull().sum() / len(df_clean)
    cols_to_drop = na_ratio[na_ratio > drop_na_threshold].index.tolist()
    
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        logger.info(f"Colonnes supprimées (>{drop_na_threshold*100}% NA): {cols_to_drop}")
    
    logger.info(f"Shape finale: {df_clean.shape}")
    return df_clean


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Gère les valeurs manquantes selon la stratégie spécifiée.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec valeurs manquantes.
    strategy : str
        Stratégie d'imputation: 'mean', 'median', 'mode', 'drop'.
    columns : Optional[List[str]]
        Colonnes à traiter. Si None, traite toutes les colonnes.
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec valeurs manquantes traitées.
    """
    df_filled = df.copy()
    
    if columns is None:
        columns = df_filled.columns.tolist()
    
    for col in columns:
        if df_filled[col].isnull().sum() > 0:
            if strategy == "mean" and df_filled[col].dtype in ["float64", "int64"]:
                df_filled[col].fillna(df_filled[col].mean(), inplace=True)
            elif strategy == "median" and df_filled[col].dtype in ["float64", "int64"]:
                df_filled[col].fillna(df_filled[col].median(), inplace=True)
            elif strategy == "mode":
                df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
            elif strategy == "drop":
                df_filled = df_filled.dropna(subset=[col])
    
    logger.info(f"Valeurs manquantes traitées avec stratégie '{strategy}'")
    return df_filled


def split_features_target(
    df: pd.DataFrame,
    target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Sépare les features et la variable cible.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame complet.
    target_column : str
        Nom de la colonne cible.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        X (features) et y (target).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    logger.info(f"Features: {X.shape[1]} colonnes, Target: '{target_column}'")
    return X, y

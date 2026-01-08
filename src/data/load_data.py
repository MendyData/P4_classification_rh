"""
=============================================================================
Module de chargement des données
=============================================================================

Ce module fournit des fonctions pour charger les données depuis 
différentes sources (CSV, Excel, bases de données, etc.)
"""

import pandas as pd
from pathlib import Path
from typing import Union, Optional
from loguru import logger


def load_raw_data(
    filepath: Union[str, Path],
    **kwargs
) -> pd.DataFrame:
    """
    Charge les données brutes depuis un fichier.
    
    Parameters
    ----------
    filepath : Union[str, Path]
        Chemin vers le fichier de données.
    **kwargs : dict
        Arguments supplémentaires passés à pd.read_csv ou pd.read_excel.
    
    Returns
    -------
    pd.DataFrame
        DataFrame contenant les données chargées.
    
    Examples
    --------
    >>> df = load_raw_data("data/raw/dataset.csv")
    >>> df.head()
    """
    filepath = Path(filepath)
    logger.info(f"Chargement des données depuis {filepath}")
    
    # Détection automatique du format
    if filepath.suffix == ".csv":
        df = pd.read_csv(filepath, **kwargs)
    elif filepath.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath, **kwargs)
    elif filepath.suffix == ".parquet":
        df = pd.read_parquet(filepath, **kwargs)
    else:
        raise ValueError(f"Format non supporté: {filepath.suffix}")
    
    logger.info(f"Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def load_processed_data(
    filename: str,
    data_dir: Union[str, Path] = "data/processed"
) -> pd.DataFrame:
    """
    Charge les données traitées depuis le dossier processed.
    
    Parameters
    ----------
    filename : str
        Nom du fichier à charger.
    data_dir : Union[str, Path]
        Répertoire des données traitées.
    
    Returns
    -------
    pd.DataFrame
        DataFrame contenant les données traitées.
    """
    filepath = Path(data_dir) / filename
    return load_raw_data(filepath)


def save_processed_data(
    df: pd.DataFrame,
    filename: str,
    data_dir: Union[str, Path] = "data/processed"
) -> Path:
    """
    Sauvegarde les données traitées.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à sauvegarder.
    filename : str
        Nom du fichier de sortie.
    data_dir : Union[str, Path]
        Répertoire de destination.
    
    Returns
    -------
    Path
        Chemin du fichier sauvegardé.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = data_dir / filename
    
    if filepath.suffix == ".csv":
        df.to_csv(filepath, index=False)
    elif filepath.suffix == ".parquet":
        df.to_parquet(filepath, index=False)
    else:
        df.to_csv(filepath, index=False)
    
    logger.info(f"Données sauvegardées: {filepath}")
    return filepath

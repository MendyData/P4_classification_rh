"""
=============================================================================
Fonctions utilitaires
=============================================================================

Collection de fonctions helper réutilisables dans tout le projet.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union
from loguru import logger
import sys


def load_config(config_path: Union[str, Path] = "configs/config.yaml") -> Dict[str, Any]:
    """
    Charge la configuration depuis un fichier YAML.
    
    Parameters
    ----------
    config_path : Union[str, Path]
        Chemin vers le fichier de configuration.
    
    Returns
    -------
    Dict[str, Any]
        Configuration sous forme de dictionnaire.
    """
    config_path = Path(config_path)
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration chargée depuis {config_path}")
    return config


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "logs/app.log"
) -> None:
    """
    Configure le système de logging avec loguru.
    
    Parameters
    ----------
    log_level : str
        Niveau de log (DEBUG, INFO, WARNING, ERROR).
    log_file : str
        Chemin du fichier de log.
    """
    # Supprimer le handler par défaut
    logger.remove()
    
    # Ajouter handler console
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=log_level
    )
    
    # Ajouter handler fichier
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_file,
        rotation="10 MB",
        retention="30 days",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
    )
    
    logger.info(f"Logging configuré - Niveau: {log_level}, Fichier: {log_file}")


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Crée un répertoire s'il n'existe pas.
    
    Parameters
    ----------
    path : Union[str, Path]
        Chemin du répertoire.
    
    Returns
    -------
    Path
        Chemin du répertoire créé.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

"""
=============================================================================
Module d'entraînement des modèles
=============================================================================

Ce module contient les fonctions pour entraîner et évaluer les modèles ML.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
from loguru import logger


def train_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "model"
) -> Any:
    """
    Entraîne un modèle de classification.
    
    Parameters
    ----------
    model : Any
        Instance du modèle sklearn.
    X_train : pd.DataFrame
        Features d'entraînement.
    y_train : pd.Series
        Labels d'entraînement.
    model_name : str
        Nom du modèle pour le logging.
    
    Returns
    -------
    Any
        Modèle entraîné.
    """
    logger.info(f"Entraînement du modèle {model_name}...")
    
    start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"Modèle {model_name} entraîné en {training_time:.2f}s")
    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model"
) -> Dict[str, float]:
    """
    Évalue les performances d'un modèle.
    
    Parameters
    ----------
    model : Any
        Modèle entraîné.
    X_test : pd.DataFrame
        Features de test.
    y_test : pd.Series
        Labels de test.
    model_name : str
        Nom du modèle.
    
    Returns
    -------
    Dict[str, float]
        Dictionnaire des métriques.
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted")
    }
    
    logger.info(f"\n=== Métriques pour {model_name} ===")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    
    return metrics


def save_model(
    model: Any,
    filepath: str,
    metadata: Optional[Dict] = None
) -> Path:
    """
    Sauvegarde un modèle entraîné.
    
    Parameters
    ----------
    model : Any
        Modèle à sauvegarder.
    filepath : str
        Chemin de sauvegarde.
    metadata : Optional[Dict]
        Métadonnées à inclure.
    
    Returns
    -------
    Path
        Chemin du fichier sauvegardé.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        "model": model,
        "metadata": metadata or {},
        "saved_at": datetime.now().isoformat()
    }
    
    with open(filepath, "wb") as f:
        pickle.dump(model_data, f)
    
    logger.info(f"Modèle sauvegardé: {filepath}")
    return filepath


def load_model(filepath: str) -> Tuple[Any, Dict]:
    """
    Charge un modèle sauvegardé.
    
    Parameters
    ----------
    filepath : str
        Chemin du modèle.
    
    Returns
    -------
    Tuple[Any, Dict]
        Modèle et ses métadonnées.
    """
    with open(filepath, "rb") as f:
        model_data = pickle.load(f)
    
    logger.info(f"Modèle chargé: {filepath}")
    return model_data["model"], model_data.get("metadata", {})

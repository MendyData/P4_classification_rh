"""
Module de gestion des données.

Ce module contient les fonctions pour:
    - Charger les données depuis différentes sources
    - Nettoyer et prétraiter les données
    - Sauvegarder les données traitées
"""

from .load_data import load_raw_data, load_processed_data
from .preprocess import clean_data, handle_missing_values

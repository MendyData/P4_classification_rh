"""
=============================================================================
Configuration pytest - Fixtures partagées
=============================================================================
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_dataframe():
    """Fixture qui retourne un DataFrame de test."""
    np.random.seed(42)
    return pd.DataFrame({
        "feature_1": np.random.randn(100),
        "feature_2": np.random.randn(100),
        "feature_3": np.random.choice(["A", "B", "C"], 100),
        "target": np.random.choice([0, 1], 100)
    })


@pytest.fixture
def project_root():
    """Fixture qui retourne le chemin racine du projet."""
    return Path(__file__).parent.parent


@pytest.fixture
def config_path(project_root):
    """Fixture qui retourne le chemin de la configuration."""
    return project_root / "configs" / "config.yaml"

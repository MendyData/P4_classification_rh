"""
=============================================================================
Tests unitaires pour le module data
=============================================================================
"""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocess import clean_data, handle_missing_values


class TestCleanData:
    """Tests pour la fonction clean_data."""
    
    def test_drop_duplicates(self, sample_dataframe):
        """Test la suppression des doublons."""
        # Créer des doublons
        df_with_dups = pd.concat([sample_dataframe, sample_dataframe.iloc[:5]])
        
        result = clean_data(df_with_dups, drop_duplicates=True)
        
        assert len(result) == len(sample_dataframe)
    
    def test_drop_na_columns(self, sample_dataframe):
        """Test la suppression des colonnes avec trop de NA."""
        df = sample_dataframe.copy()
        df["mostly_na"] = np.nan  # Colonne 100% NA
        
        result = clean_data(df, drop_na_threshold=0.5)
        
        assert "mostly_na" not in result.columns


class TestHandleMissingValues:
    """Tests pour la fonction handle_missing_values."""
    
    def test_median_imputation(self, sample_dataframe):
        """Test l'imputation par médiane."""
        df = sample_dataframe.copy()
        df.loc[0:10, "feature_1"] = np.nan
        
        result = handle_missing_values(df, strategy="median")
        
        assert result["feature_1"].isnull().sum() == 0

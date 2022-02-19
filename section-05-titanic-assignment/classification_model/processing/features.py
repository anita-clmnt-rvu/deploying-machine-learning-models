from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ExtractLetterTransformer(BaseEstimator, TransformerMixin):
    """Extract fist letter of variable."""

    def __init__(self, variables: List[str]) -> pd.DataFrame:
        """Initialises the ExtractLetterTransformer class."""
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series=None):
        """Fit the sklearn pipeline."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the original pandas Dataframe."""
        X = X.copy()
        
        for feature in self.variables:
            X[feature] = X[feature].str[0]

        return X
# File: src/models/base_model.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseMLModel(ABC):
    """Base class for all machine learning models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the model."""
        pass

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """Train the model on the provided data."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> pd.Series:
        """Make predictions on the provided data."""
        pass

    @abstractmethod
    def tune(self, X_train: pd.DataFrame, y_train: pd.Series, 
             param_grid: dict, k_folds: int, scalers: dict, **kwargs) -> pd.DataFrame:
        """Tune hyperparameters and return a sorted dataframe of results."""
        pass

    @abstractmethod
    def get_default_param_grid(self, X_train) -> dict:
        """Return the default hyperparameter grid for the model."""
        pass
    
    @abstractmethod
    def get_param_definitions(self) -> dict:
        """Return the parameter definitions for UI configuration."""
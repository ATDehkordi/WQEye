# File: src/models/registry.py
from src.models.ann import ANNModel
from src.models.kan import KANModel
from src.models.base_model import BaseMLModel
from src.models.random_forest import RandomForestModel

class ModelRegistry:
    """Simple registry to manage available ML models."""

    def __init__(self):
        # Manually define available models
        self.models = {
            "Random Forest": RandomForestModel(),
            "Artificial Neural Network": ANNModel(),
            "Kolmogrov-Arnold Network": KANModel()
        }

    def get_model(self, model_name: str) -> BaseMLModel:
        """Get a model instance by name."""
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found in registry.")
        return model

    def get_model_names(self) -> list:
        """Return list of available model names."""
        return list(self.models.keys())
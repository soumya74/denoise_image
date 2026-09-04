# models/__init__.py

class ModelRegistry:
    def __init__(self):
        self._models = {}

    def register(self, name):
        def decorator(cls):
            self._models[name.lower()] = cls
            return cls
        return decorator

    def build(self, name, **kwargs):
        name = name.lower()
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry. Available: {list(self._models.keys())}")
        return self._models[name](**kwargs)

MODEL_REGISTRY = ModelRegistry()

# --- Model Imports (Must remain below MODEL_REGISTRY) ---
from .baseline_cnn import BaselineDenoiser
from .dncnn_model import DnCNN
from .disentangle_net import DisentanglementNetwork
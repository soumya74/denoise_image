import torch.nn as nn
from .disentangle_net import DisentanglementNetwork

class ModelRegistry:
    def __init__(self):
        self._models = {}

    def register(self, name):
        def decorator(cls):
            self._models[name.lower()] = cls
            return cls
        return decorator

    def build(self, name, **kwargs) -> nn.Module:
        name = name.lower()
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found. Available models: {list(self._models.keys())}")
        return self._models[name](**kwargs)

MODEL_REGISTRY = ModelRegistry()

# Automatically import all model files in this folder to register them
from .base_model import BaselineDenoiser
try:
    from .dncnn_model import DnCNN
except ImportError:
    pass
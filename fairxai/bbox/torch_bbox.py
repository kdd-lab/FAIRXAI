import os
from typing import Optional, Type

import torch

from fairxai.bbox import AbstractBBox
from fairxai.logger import logger


class TorchBBox(AbstractBBox):
    """
    Wrapper for PyTorch models to unify interface with AbstractBBox.
    Stores logical model type for explainer compatibility.
    """

    def __init__(self, model=None, model_type: str = None, model_name: str = None, device: str = "cpu"):
        """
        Args:
            model: PyTorch nn.Module instance
            model_type: logical model type (e.g., 'torch_mlp')
            model_name: optional name for tracking
            device: device for inference ('cpu' or 'cuda')
        """
        super().__init__()
        self.model = model
        self.model_type = model_type or "torch_generic"
        self.framework = "torch"
        self.device = device
        self.model_name = model_name

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def predict(self, X):
        """Return class predictions"""
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            return torch.argmax(logits, dim=1).cpu().numpy()

    def predict_proba(self, X):
        """Return probabilities"""
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def load(self, path: str, model_cls: Optional[Type[torch.nn.Module]] = None):
        """
        Load PyTorch model weights.

        Args:
            path: file path to .pth file
            model_cls: optional class to instantiate
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file '{path}' not found")
        if self.model is None:
            if model_cls is None:
                raise ValueError("model_cls must be provided to instantiate a PyTorch model")
            self.model = model_cls()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"TorchBBox model loaded from {path}")

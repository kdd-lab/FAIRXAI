from typing import Dict, Any, Type, Optional
from sklearn.base import BaseEstimator
import torch.nn as nn

from fairxai.bbox import AbstractBBox
from fairxai.bbox.sklearn_bbox import SklearnBBox
from fairxai.bbox.torch_bbox import TorchBBox


class ModelFactory:
    """
    Scalable factory for creating AbstractBBox wrappers (SklearnBBox, TorchBBox, etc.)
    based on the framework name. Supports dynamic framework registration.

    This approach avoids the need for a model instance and works directly with
    saved files (.pkl/.pth) by calling the wrapper's `load()` method.
    """

    # Registry: framework -> wrapper class + base class for validation
    _framework_registry: Dict[str, Dict[str, Any]] = {
        "sklearn": {"wrapper": SklearnBBox, "class": BaseEstimator},
        "torch": {"wrapper": TorchBBox, "class": nn.Module},
    }

    @classmethod
    def create(
            cls,
            framework: str,
            model_path: Optional[str] = None,
            model_params: Optional[Dict[str, Any]] = None,
            device: str = "cpu"
    ) -> AbstractBBox:
        """
        Instantiate the correct AbstractBBox wrapper for the given framework.

        Args:
            framework: Name of the ML framework ('sklearn', 'torch', etc.)
            model_path: Optional path to pre-trained model (.pkl or .pth)
            model_params: Optional parameters to initialize the model if needed
            device: Device for TorchBBox ('cpu' or 'cuda')

        Returns:
            AbstractBBox instance with loaded model if model_path provided

        Raises:
            ValueError: if framework is unsupported or wrapper instantiation fails
        """
        framework = framework.lower()
        if framework not in cls._framework_registry:
            raise ValueError(f"Unsupported framework '{framework}'. "
                             f"Available: {list(cls._framework_registry.keys())}")

        wrapper_cls = cls._framework_registry[framework]["wrapper"]

        # Instantiate the wrapper
        if issubclass(wrapper_cls, SklearnBBox):
            bbox = wrapper_cls(model=None)  # model will be loaded if model_path is provided
        elif issubclass(wrapper_cls, TorchBBox):
            bbox = wrapper_cls(model=None, device=device)
        else:
            raise ValueError(f"Unsupported wrapper class {wrapper_cls}")

        # Load model if path provided
        if model_path:
            bbox.load(model_path)

        return bbox

    @classmethod
    def register_framework(cls, framework_name: str, wrapper_cls: Type[AbstractBBox], base_class: Type):
        """
        Dynamically register a new ML framework.

        Args:
            framework_name: Logical name of the framework
            wrapper_cls: Wrapper class implementing AbstractBBox
            base_class: Base class for validation (optional)
        """
        cls._framework_registry[framework_name.lower()] = {"wrapper": wrapper_cls, "class": base_class}

    @classmethod
    def available_frameworks(cls) -> list[str]:
        """Return all currently registered frameworks."""
        return list(cls._framework_registry.keys())

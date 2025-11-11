from typing import Dict, Any, Type, Optional

from fairxai.bbox import AbstractBBox
from fairxai.bbox.sklearn_bbox import SklearnBBox
from fairxai.bbox.torch_bbox import TorchBBox


class ModelFactory:
    """
    Factory for creating AbstractBBox wrappers (e.g., SklearnBBox, TorchBBox)
    and managing registration of model types dynamically.

    This factory supports pre-registered model types and allows runtime
    extension for custom or user-defined models.
    """

    _model_registry: Dict[str, Dict[str, Any]] = {
        # -------------------------
        # Scikit-learn models
        # -------------------------
        "sklearn_tree": {"wrapper": SklearnBBox, "framework": "sklearn"},
        "sklearn_random_forest": {"wrapper": SklearnBBox, "framework": "sklearn"},
        "sklearn_gradient_boosting": {"wrapper": SklearnBBox, "framework": "sklearn"},
        "sklearn_linear": {"wrapper": SklearnBBox, "framework": "sklearn"},
        "sklearn_logistic": {"wrapper": SklearnBBox, "framework": "sklearn"},
        "sklearn_svm": {"wrapper": SklearnBBox, "framework": "sklearn"},
        "sklearn_knn": {"wrapper": SklearnBBox, "framework": "sklearn"},
        "sklearn_pls": {"wrapper": SklearnBBox, "framework": "sklearn"},

        # -------------------------
        # PyTorch models
        # -------------------------
        "torch_mlp": {"wrapper": TorchBBox, "framework": "torch"},
        "torch_cnn": {"wrapper": TorchBBox, "framework": "torch"},
        "torch_rnn": {"wrapper": TorchBBox, "framework": "torch"},
        "torch_lstm": {"wrapper": TorchBBox, "framework": "torch"},
        "torch_gru": {"wrapper": TorchBBox, "framework": "torch"},
        "torch_generic": {"wrapper": TorchBBox, "framework": "torch"},
        "torch_transformer": {"wrapper": TorchBBox, "framework": "torch"},
    }

    # ===============================================================
    # Factory creation
    # ===============================================================
    @classmethod
    def create(
        cls,
        model_type: str,
        model_name: Optional[str] = None,
        model_instance: Any = None,
        model_params: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ) -> AbstractBBox:
        """
        Instantiate a black-box model wrapper with the correct logical type and framework.

        Args:
            model_type: Logical model type (must exist in _model_registry)
            model_name: Optional name for tracking / registry
            model_instance: Optional pre-trained model instance
            model_params: Optional dictionary of parameters to initialize model
            model_path: Optional file path to load pre-trained model
            device: Optional device for TorchBBox ('cpu' or 'cuda')

        Returns:
            AbstractBBox instance with model_type and framework set
        """
        key = model_type.lower()
        if key not in cls._model_registry:
            raise ValueError(
                f"Unknown model_type '{model_type}'. Supported: {list(cls._model_registry.keys())}"
            )

        wrapper_cls = cls._model_registry[key]["wrapper"]

        # -----------------------
        # Wrapper initialization
        # -----------------------
        if issubclass(wrapper_cls, SklearnBBox):
            bbox = wrapper_cls(model=model_instance, model_type=key, model_name=model_name)
        elif issubclass(wrapper_cls, TorchBBox):
            bbox = wrapper_cls(model=model_instance, model_type=key, model_name=model_name, device=device)
        else:
            raise ValueError(f"Unsupported wrapper class '{wrapper_cls.__name__}'")

        # -----------------------
        # Optional loading
        # -----------------------
        if model_path:
            if isinstance(bbox, SklearnBBox):
                bbox.load(model_path)
            elif isinstance(bbox, TorchBBox):
                if model_instance is None and model_params and "model_cls" in model_params:
                    bbox.load(model_path, model_cls=model_params["model_cls"])
                else:
                    bbox.load(model_path)

        return bbox

    # ===============================================================
    # Dynamic registration
    # ===============================================================
    @classmethod
    def register_model(
        cls,
        model_type: str,
        wrapper_cls: Type[AbstractBBox],
        framework: str,
        overwrite: bool = False,
    ):
        """
        Dynamically register a new model type into the factory.

        Args:
            model_type: Logical identifier (e.g., 'sklearn_xgboost')
            wrapper_cls: Subclass of AbstractBBox (e.g., SklearnBBox)
            framework: Framework name ('sklearn', 'torch', etc.)
            overwrite: If True, overwrites an existing entry.

        Example:
            ModelFactory.register_model(
                model_type="sklearn_xgboost",
                wrapper_cls=SklearnBBox,
                framework="sklearn"
            )
        """
        key = model_type.lower()
        if key in cls._model_registry and not overwrite:
            raise ValueError(f"Model type '{key}' already exists. Use overwrite=True to replace.")
        cls._model_registry[key] = {"wrapper": wrapper_cls, "framework": framework}

    # ===============================================================
    # Utility
    # ===============================================================
    @classmethod
    def available_models(cls) -> list[str]:
        """Returns a list of all registered model names."""
        return list(cls._model_registry.keys())

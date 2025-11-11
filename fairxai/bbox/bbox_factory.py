from typing import Dict, Any

from fairxai.bbox import AbstractBBox
from fairxai.bbox.sklearn_bbox import SklearnBBox
from fairxai.bbox.torch_bbox import TorchBBox


class ModelFactory:
    """
    Factory to instantiate black-box models (AbstractBBox wrappers)
    with framework and logical model type metadata.

    The _model_registry maps a logical model type to the corresponding wrapper
    class and the framework it belongs to.
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

        # -------------------------
        # PyTorch models
        # -------------------------
        "torch_mlp": {"wrapper": TorchBBox, "framework": "torch"},
        "torch_cnn": {"wrapper": TorchBBox, "framework": "torch"},
        "torch_rnn": {"wrapper": TorchBBox, "framework": "torch"},
        "torch_lstm": {"wrapper": TorchBBox, "framework": "torch"},
        "torch_gru": {"wrapper": TorchBBox, "framework": "torch"},
        "torch_generic": {"wrapper": TorchBBox, "framework": "torch"},
        "torch_transformer": {"wrapper": TorchBBox, "framework": "torch"}
    }

    @classmethod
    def create(cls,
               model_type: str,
               model_name: str = None,
               model_instance: Any = None,
               model_params: Dict[str, Any] = None,
               model_path: str = None,
               device: str = "cpu") -> AbstractBBox:
        """
        Instantiate a black-box model wrapper with the correct logical type and framework.

        Args:
            model_type: logical model type (must exist in _model_registry)
            model_name: optional name for tracking / registry
            model_instance: optional pre-trained model instance
            model_params: optional dictionary of parameters to initialize model
            model_path: optional file path to load pre-trained model
            device: optional device for TorchBBox ('cpu' or 'cuda')

        Returns:
            AbstractBBox instance with model_type and framework set
        """
        key = model_type.lower()
        if key not in cls._model_registry:
            raise ValueError(
                f"Unknown model_type '{model_type}'. Supported: {list(cls._model_registry.keys())}"
            )

        wrapper_cls = cls._model_registry[key]["wrapper"]

        # Instantiate wrapper
        if issubclass(wrapper_cls, SklearnBBox):
            bbox = wrapper_cls(model=model_instance, model_type=key, model_name=model_name)
        elif issubclass(wrapper_cls, TorchBBox):
            bbox = wrapper_cls(model=model_instance, model_type=key, model_name=model_name, device=device)
        else:
            raise ValueError(f"Unsupported wrapper class '{wrapper_cls.__name__}'")

        # Load the pre-trained model if a path is provided
        if model_path:
            if isinstance(bbox, SklearnBBox):
                bbox.load(model_path)
            elif isinstance(bbox, TorchBBox):
                # If model_instance is None, must provide model_cls in model_params
                if model_instance is None and model_params and "model_cls" in model_params:
                    bbox.load(model_path, model_cls=model_params["model_cls"])
                else:
                    bbox.load(model_path)

        return bbox

    # ===============================================================
    # Utility
    # ===============================================================
    @classmethod
    def available_models(cls) -> list[str]:
        """Returns a list of all registered model names."""
        return list(cls._model_registry.keys())

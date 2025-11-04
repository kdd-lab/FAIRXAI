from typing import Type, Optional, Dict, Any

from fairxai.bbox import AbstractBBox


class ModelFactory:
    """
    Factory responsible for creating model (black-box) instances
    based on a registry mapping model names to concrete subclasses
    of `AbstractBBox`.

    This class enables dynamic instantiation of models by name,
    allowing flexible extension through runtime registration.
    """

    # Registry maps lowercase names to class references (not strings)
    _model_registry: Dict[str, Type[AbstractBBox]] = {}

    # ===============================================================
    # Registration
    # ===============================================================
    @classmethod
    def register(cls, name: str, model_class: Type[AbstractBBox]):
        """
        Registers a new model class under a given name.

        Args:
            name: The key name identifying the model (case-insensitive).
            model_class: The concrete subclass of AbstractBBox to register.

        Raises:
            TypeError: If model_class is not a subclass of AbstractBBox.
        """
        if not issubclass(model_class, AbstractBBox):
            raise TypeError(f"{model_class.__name__} must inherit from AbstractBBox")

        key = name.lower()
        cls._model_registry[key] = model_class

    # ===============================================================
    # Creation
    # ===============================================================
    @classmethod
    def create(cls, model_name: str, model_params: Optional[Dict[str, Any]] = None) -> AbstractBBox:
        """
        Instantiates a registered model by name, passing optional parameters.

        Args:
            model_name: The string name of the model (case-insensitive).
            model_params: Optional dictionary of initialization parameters.

        Returns:
            An instance of the requested model subclass.

        Raises:
            ValueError: If the model_name is not registered.
        """
        key = model_name.lower()
        if key not in cls._model_registry:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Registered models: {list(cls._model_registry.keys())}"
            )

        model_cls = cls._model_registry[key]
        return model_cls(**(model_params or {}))

    # ===============================================================
    # Utility
    # ===============================================================
    @classmethod
    def available_models(cls) -> list[str]:
        """Returns a list of all registered model names."""
        return list(cls._model_registry.keys())

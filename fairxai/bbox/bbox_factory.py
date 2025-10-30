from typing import Type, Optional, Dict, Any

from fairxai.bbox import AbstractBBox


class ModelFactory:
    """
    Factory to instantiate models by name.
    """

    _model_registry: dict[str, Type[AbstractBBox]] = {
        "mybboxmodela": "MyBBoxModelA"  # FIXME: qui andrà l'istanza della sottoclasse di AbstractBBox specifica
    }

    @classmethod
    def create(cls, model_name: str, model_params: Optional[Dict[str, Any]]) -> AbstractBBox:
        """
        Creates a model instance given its name and optional parameters.

        Args:
            model_name: The string name of the model to instantiate.
            model_params: Optional dictionary of parameters to pass to the model.

        Returns:
            An instance of the requested model.

        Raises:
            ValueError if the model_name is not registered.
        """
        key = model_name.lower()
        if key not in cls._model_registry:
            raise ValueError(f"Unknown model '{model_name}'")

        model_cls = cls._model_registry[key]
        return model_cls(model_params)

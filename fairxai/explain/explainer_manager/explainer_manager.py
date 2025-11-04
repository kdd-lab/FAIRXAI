from typing import List, Type

from fairxai.explain.adapter.generic_explainer_adapter import GenericExplainerAdapter
from fairxai.logger import logger


class ExplainerManager:
    """
    Manages explainer discovery, compatibility filtering, and instantiation.

    This version initializes with a specific dataset type and model name,
    loads only compatible explainers at construction, and provides a clean
    factory method to instantiate one by name.
    """

    def __init__(self, dataset_type: str, model_name: str):
        """
        Initializes the manager for a specific dataset type and model name.

        Args:
            dataset_type: The type of dataset (e.g., 'tabular', 'image', 'text').
            model_name: The name of the model (e.g., 'MyBBoxModelA').
        """
        self.dataset_type, self.model_name = self._normalize_type_names(dataset_type, model_name)
        self.explainers: dict[str, Type[GenericExplainerAdapter]] = {}
        self._load_compatible_explainers()

    def _load_compatible_explainers(self) -> None:
        """
        Discovers and stores only explainer classes compatible with
        the given dataset_type and model_name.
        """
        all_classes = self._get_all_explainer_classes()
        compatible = []

        for cls in all_classes:
            try:
                if cls.is_compatible(self.dataset_type, self.model_name):
                    compatible.append(cls)
            except Exception as e:
                logger.warning(f"Error checking compatibility for {cls.__name__}: {e}")

        self.explainers = {cls.__name__: cls for cls in compatible}
        logger.info(f"Loaded {len(self.explainers)} compatible explainers for "
                    f"{self.dataset_type} + {self.model_name}")

    def list_available_compatible_explainers(self) -> List[Type[GenericExplainerAdapter]]:
        """
        Returns all explainer classes compatible with the current dataset/model pair.
        """
        return list(self.explainers.values())

    def create_explainer(self, explainer_name: str, model_instance, dataset_instance) -> GenericExplainerAdapter:
        """
        Instantiates a single explainer by name using provided model and dataset instances.

        Args:
            explainer_name: The name of the explainer class to instantiate.
            model_instance: The actual model object to be explained.
            dataset_instance: The dataset instance used for the explainer.

        Returns:
            An initialized explainer instance.

        Raises:
            ValueError if the explainer is not compatible or not found.
        """
        if explainer_name not in self.explainers:
            available = list(self.explainers.keys())
            raise ValueError(
                f"No compatible explainer named '{explainer_name}' found "
                f"for dataset '{self.dataset_type}' and model '{self.model_name}'. "
                f"Available explainers: {available}"
            )

        explainer_cls = self.explainers[explainer_name]
        logger.info(f"Creating explainer '{explainer_name}' for "
                    f"{self.dataset_type} + {self.model_name}")
        return explainer_cls(model=model_instance, dataset=dataset_instance)

    # ============================================================
    # INTERNAL UTILITIES
    # ============================================================
    @staticmethod
    def _get_all_explainer_classes() -> List[Type[GenericExplainerAdapter]]:
        """Recursively retrieve all subclasses (direct and indirect)
        of GenericExplainerAdapter currently loaded."""

        def recurse(cls):
            subclasses = []
            for sub in cls.__subclasses__():
                subclasses.append(sub)
                subclasses.extend(recurse(sub))
            return subclasses

        return recurse(GenericExplainerAdapter)

    @staticmethod
    def _normalize_type_names(*names: str) -> tuple:
        """Normalize type names to lowercase for a consistent comparison."""
        return tuple(name.lower() for name in names)

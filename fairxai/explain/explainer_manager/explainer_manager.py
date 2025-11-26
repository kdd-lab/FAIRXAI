import pkgutil
import importlib
from pathlib import Path
from typing import List, Type

import fairxai.explain.adapter as adapter

from fairxai.explain.adapter.generic_explainer_adapter import GenericExplainerAdapter
from fairxai.logger import logger


def import_all_adapters(package):
    """
    Dynamically imports all modules in a package and its subpackages
    so that all subclasses of GenericExplainerAdapter are loaded.
    """
    package_path = Path(package.__file__).parent
    for _, module_name, is_pkg in pkgutil.walk_packages([str(package_path)], prefix=package.__name__ + "."):
        importlib.import_module(module_name)
        if is_pkg:
            subpackage = importlib.import_module(module_name)
            import_all_adapters(subpackage)


class ExplainerManager:
    """
    Manages explainer discovery, compatibility filtering, and instantiation.

    Auto-discovers all adapters in the `adapter` package, filters
    those compatible with the dataset type and model type, and allows
    easy instantiation of explainers by name.
    """

    def __init__(self, dataset_type: str, model_type: str):
        """
        Initialize the manager for a specific dataset type and model type.

        Args:
            dataset_type: Type of dataset (e.g., 'tabular', 'image', 'text').
            model_type: Type of model (class name, e.g., 'SklearnRandomForestClassifier', 'XGBoostClassifier').
        """
        self.dataset_type, self.model_type = dataset_type, model_type

        # Import all adapters so subclasses are registered
        import_all_adapters(adapter)

        # Load compatible explainers
        self.explainers: dict[str, Type[GenericExplainerAdapter]] = {}
        self._load_compatible_explainers()

    def _load_compatible_explainers(self):
        """
        Discover all subclasses of GenericExplainerAdapter and store
        only those compatible with the current dataset and model.
        """
        all_classes = self._get_all_explainer_classes()
        compatible = []

        for cls in all_classes:
            try:
                if cls.is_compatible(self.dataset_type, self.model_type):
                    compatible.append(cls)
            except Exception as e:
                logger.warning(f"Error checking compatibility for {cls.__name__}: {e}")

        self.explainers = {cls.__name__: cls for cls in compatible}
        logger.info(f"Loaded {len(self.explainers)} compatible explainers for "
                    f"{self.dataset_type} + {self.model_type}: {list(self.explainers.keys())}")

    def list_available_compatible_explainers(self) -> List[Type[GenericExplainerAdapter]]:
        """Return all explainer classes compatible with the current dataset/model."""
        return list(self.explainers.values())

    def create_explainer(self, explainer_name: str, model_instance, dataset_instance) -> GenericExplainerAdapter:
        """
        Instantiate an explainer by name using the provided model and dataset.

        Args:
            explainer_name: Name of the explainer class.
            model_instance: The trained model object.
            dataset_instance: Dataset object.

        Returns:
            An initialized explainer instance.

        Raises:
            ValueError if explainer not found or not compatible.
        """
        if explainer_name not in self.explainers:
            available = list(self.explainers.keys())
            raise ValueError(
                f"No compatible explainer named '{explainer_name}' found "
                f"for dataset '{self.dataset_type}' and model '{self.model_type}'. "
                f"Available explainers: {available}"
            )

        explainer_cls = self.explainers[explainer_name]
        logger.info(f"Creating explainer '{explainer_name}' for "
                    f"{self.dataset_type} + {self.model_type}")
        return explainer_cls(model=model_instance, dataset=dataset_instance)

    # ============================================================
    # INTERNAL UTILITIES
    # ============================================================

    @staticmethod
    def _get_all_explainer_classes() -> List[Type[GenericExplainerAdapter]]:
        """Return all subclasses (direct and indirect) of GenericExplainerAdapter."""
        def recurse(cls):
            subclasses = []
            for sub in cls.__subclasses__():
                subclasses.append(sub)
                subclasses.extend(recurse(sub))
            return subclasses

        return recurse(GenericExplainerAdapter)

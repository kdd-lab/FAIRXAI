from typing import Any, Dict, Optional

import numpy as np
from lore_sa import (
    TabularGeneticGeneratorLore,
    TabularRandomGeneratorLore,
    TabularRandGenGeneratorLore
)

from fairxai.bbox import AbstractBBox
from fairxai.data.dataset import TabularDataset
from fairxai.explain.adapter.generic_explainer_adapter import GenericExplainerAdapter
from fairxai.explain.explaination.generic_explanation import GenericExplanation
from fairxai.logger import logger


class LoreExplainerAdapter(GenericExplainerAdapter):
    """
    LoreExplainerAdapter: Lazy Initialization and Dynamic Runtime Configuration

    This adapter wraps the LORE (Local Rule-based Explanation) algorithm for tabular datasets.
    It implements lazy initialization: the actual LORE explainer is created only when an
    explanation is requested (via `explain_instance` or `explain_global`). This design allows
    dynamic configuration of runtime parameters and keeps pipeline integration simple and efficient.
    
    Key Features:
    - Lazy Initialization: The explainer is instantiated only when needed, avoiding
      unnecessary overhead and premature dependency loading.
    - Runtime Configurability: Explanation strategies ("genetic", "random", "hybrid")
      and parameters like `num_samples` can be supplied per call, without reconstructing the adapter.
    - Pipeline-friendly: Explainer factories/managers do not need prior knowledge of
      specific strategies or parameters; the adapter handles setup internally.
    - Compatible with all local/global explanation modes for tabular datasets.
    """

    supported_datasets = ["tabular"]
    supported_models = [
        "sklearn_tree",
        "sklearn_random_forest",
        "sklearn_gradient_boosting",
        "sklearn_linear",
        "sklearn_logistic",
        "sklearn_svm",
        "torch_mlp",
        "torch_generic",
    ]

    DEFAULT_STRATEGY = "genetic"

    def __init__(self, model: AbstractBBox, dataset: TabularDataset):
        """
        Initialize the adapter (explainer will be created lazily when needed).
        """
        super().__init__(model, dataset)
        self.explainer = None
        self.strategy = None  # will be set at runtime

    # --------------------------------------------------------
    # Lazy initialization
    # --------------------------------------------------------
    def _init_explainer(self, strategy: str):
        """Initialize the LORE explainer for a specific strategy."""
        strategy = strategy.lower()
        try:
            explainer_cls = {
                "genetic": TabularGeneticGeneratorLore,
                "random": TabularRandomGeneratorLore,
                "hybrid": TabularRandGenGeneratorLore,
            }.get(strategy)

            if explainer_cls is None:
                raise ValueError(f"Unknown LORE strategy '{strategy}'")

            self.explainer = explainer_cls(self.model.model, self.dataset)
            self.strategy = strategy
            logger.info(f"Initialized LORE explainer ({explainer_cls.__name__}).")
        except Exception as e:
            logger.error(f"Failed to initialize LORE explainer: {e}")
            raise

    # --------------------------------------------------------
    # Local explanation
    # --------------------------------------------------------
    def explain_instance(self, instance: Any, params: Optional[Dict[str, Any]] = None) -> GenericExplanation:
        """
        Compute a local explanation for a single dataset instance.

        Args:
            instance: The data instance to explain. Should be compatible with the dataset used.
            params: Optional dictionary of runtime parameters, e.g.,
                - "strategy": str, one of ["genetic", "random", "hybrid"] (default: "genetic")
                - "num_samples": int, number of synthetic neighborhood samples (default: 500)
                - other algorithm-specific parameters.

        Returns:
            GenericExplanation: An object containing factual rules, counterfactuals, feature importances,
            and fidelity scores for the instance.
        """
        if params is None:
            params = {}

        # Extract params
        strategy = params.get("strategy", self.DEFAULT_STRATEGY).lower()
        num_samples = params.get("num_samples", 1500)

        # Initialize explainer lazily if missing or strategy changed
        if self.explainer is None or strategy != self.strategy:
            self._init_explainer(strategy)

        # Prepare instance
        if hasattr(instance, "to_array"):
            x = np.asarray(instance.to_array()).reshape(1, -1)
        else:
            x = np.asarray(instance).reshape(1, -1)

        try:
            explanation = self.explainer.explain_instance(x[0], num_samples=num_samples)
        except Exception as e:
            logger.error(f"LORE explanation failed: {e}")
            raise

        payload = {
            "rule": str(explanation.get("rule")),
            "counterfactuals": explanation.get("counterfactuals", []),
            "feature_importances": explanation.get("feature_importances", {}),
            "fidelity": float(explanation.get("fidelity", 0.0)),
            "strategy": strategy,
        }

        return self.build_generic_explanation(
            data=payload, explanation_type=self.LOCAL_EXPLANATION
        )

    # --------------------------------------------------------
    # Global explanation
    # --------------------------------------------------------
    def explain_global(self, params: Optional[Dict[str, Any]] = None) -> GenericExplanation:
        """
        Compute a global explanation by summarizing local explanations across instances.

        Args:
            params: Optional dictionary of runtime parameters, similar to `explain_instance`.
                    Can include strategy, number of samples, or other aggregation parameters.

        Returns:
            GenericExplanation: An object containing aggregated rules, feature importances,
            and other global insights.
        """
        if params is None:
            params = {}

        strategy = params.get("strategy", self.DEFAULT_STRATEGY)
        n_samples = params.get("n_samples", 10)
        num_samples = params.get("num_samples", 1000)

        # Lazy init (again)
        if self.explainer is None or strategy != self.strategy:
            self._init_explainer(strategy)

        explanations = []
        for i in range(min(n_samples, len(self.dataset))):
            instance = self.dataset[i]
            try:
                exp = self.explain_instance(instance, {"num_samples": num_samples, "strategy": strategy})
                explanations.append(exp.to_dict())
            except Exception as e:
                logger.warning(f"Global LORE: skipped instance {i} ({e})")

        aggregated_rules = [e["data"]["rule"] for e in explanations]

        payload = {
            "aggregated_rules": aggregated_rules,
            "num_instances": len(explanations),
            "strategy": strategy,
        }

        return self.build_generic_explanation(
            data=payload, explanation_type=self.GLOBAL_EXPLANATION
        )

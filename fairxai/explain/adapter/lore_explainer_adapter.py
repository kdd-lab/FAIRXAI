from typing import Optional, Dict, Any, List
import numpy as np

from fairxai.explain.explaination.generic_explanation import GenericExplanation
from fairxai.explain.explaination.rule_based_explanation import RuleBasedExplanation
from fairxai.explain.explaination.counterfactual_rule_explanation import CounterfactualRuleExplanation
from fairxai.explain.explaination.feature_importance_explanation import FeatureImportanceExplanation

from lore_sa import TabularGeneticGeneratorLore, TabularRandomGeneratorLore, TabularRandGenGeneratorLore

from fairxai.logger import logger


class LoreExplainerAdapter:
    """
    Adapter for LORE (Local Rule-Based Explanations) explainers.

    Provides a unified interface to compute local and global explanations
    from black-box models. The adapter lazily initializes the LORE explainer
    only when needed, supporting multiple neighborhood generation strategies.

    Attributes:
        bbox: Black-box model wrapped in AbstractBBox.
        dataset: Dataset object with descriptor information.
        explainer: Instance of a LORE explainer (genetic, random, or probabilistic-genetic).
        strategy: Current neighborhood generation strategy.
    """

    DEFAULT_STRATEGY = "genetic"

    def __init__(self, bbox, dataset):
        """
        Initialize the adapter without creating the LORE explainer yet.

        Args:
            bbox: Black-box model wrapped in AbstractBBox.
            dataset: TabularDataset containing feature descriptor and data.
        """
        self.bbox = bbox
        self.dataset = dataset
        self.explainer = None
        self.strategy = None

    # -----------------------------
    # Internal helper methods
    # -----------------------------
    def _init_explainer(self, strategy: str = "genetic"):
        """
        Lazily initialize the LORE explainer based on the chosen strategy.

        Args:
            strategy: Neighborhood generation strategy. One of:
                "genetic" (default), "random", "probabilistic-genetic".
        """
        strategy = strategy.lower()
        if strategy == "genetic":
            self.explainer = TabularGeneticGeneratorLore(self.bbox, self.dataset)
        elif strategy == "random":
            self.explainer = TabularRandomGeneratorLore(self.bbox, self.dataset)
        elif strategy == "probabilistic-genetic":
            self.explainer = TabularRandGenGeneratorLore(self.bbox, self.dataset)
        else:
            raise ValueError(f"Unknown LORE strategy: {strategy}")
        self.strategy = strategy
        logger.info(f"LORE explainer initialized with strategy '{strategy}'")

    def _rule_to_predicates(self, rule_dict: dict) -> list[str]:
        """
        Convert a LORE rule dict into a list of human-readable predicates
        for RuleBasedExplanation.

        Each entry in the premise becomes its own 'IF ... THEN ...' string.
        """
        consequence = rule_dict.get("consequence", "Unknown")
        predicates = []
        for condition in rule_dict.get("premise", []):
            predicates.append(f"IF {condition} THEN class = {consequence}")
        return predicates

    def _map_lore_to_explanations(self, lore_output: dict, explainer_name: str = "LORE") -> List[GenericExplanation]:
        """
        Convert raw LORE output dictionary to concrete explanation objects.

        Args:
            lore_output: Dictionary returned by LORE explainer.
            explainer_name: Name of the explainer for tracking.

        Returns:
            List of GenericExplanation instances:
                - RuleBasedExplanation
                - CounterfactualRuleExplanation
                - FeatureImportanceExplanation
        """
        rules_dict = lore_output.get("rules", {})

        factual = RuleBasedExplanation(explainer_name, self._rule_to_predicates(rules_dict))
        counterfactuals = CounterfactualRuleExplanation(explainer_name, lore_output.get("counterfactuals", []))
        feature_importances = FeatureImportanceExplanation(explainer_name,
                                                           dict(lore_output.get("feature_importances", {})))
        return [factual, counterfactuals, feature_importances]

    # -----------------------------
    # Public methods
    # -----------------------------
    def explain_instance(self, instance: Any, params: Optional[Dict[str, Any]] = None) -> List[GenericExplanation]:
        """
        Compute a local explanation for a single instance.

        Args:
            instance: Input instance compatible with the dataset.
            params: Optional runtime parameters:
                - "strategy": str, one of ["genetic", "random", "probabilistic-genetic"]
                - "num_samples": int, number of synthetic samples (default: 1500)

        Returns:
            List of GenericExplanation objects representing factual rules, counterfactuals,
            and feature importances.
        """
        if params is None:
            params = {}
        strategy = params.get("strategy", self.DEFAULT_STRATEGY).lower()
        num_samples = params.get("num_samples", 1500)

        # Initialize explainer lazily or if strategy changed
        if self.explainer is None or strategy != self.strategy:
            self._init_explainer(strategy)

        # Ensure instance is a 1D array
        x = np.asarray(instance).reshape(1, -1) if not isinstance(instance, np.ndarray) else instance

        try:
            lore_output = self.explainer.explain_instance(x[0])
        except Exception as e:
            logger.error(f"LORE explanation failed: {e}")
            raise

        return self._map_lore_to_explanations(lore_output, explainer_name="LORE")

    def explain_global(self, instances: Any, params: Optional[Dict[str, Any]] = None) -> List[GenericExplanation]:
        """
        Compute a global explanation for a set of instances by aggregating local explanations.

        Currently, this method averages feature importances and collects unique rules
        across the instances. Counterfactuals are not aggregated.

        Args:
            instances: Iterable of input instances.
            params: Optional runtime parameters passed to explain_instance.

        Returns:
            List of GenericExplanation objects representing global rules and feature importances.
        """
        if params is None:
            params = {}

        # Aggregate rules and feature importances
        all_rules = []
        all_feature_importances = {}

        for instance in instances:
            explanations = self.explain_instance(instance, params=params)
            for exp in explanations:
                if isinstance(exp, RuleBasedExplanation):
                    all_rules.extend(exp.data.get("rules", []))
                elif isinstance(exp, FeatureImportanceExplanation):
                    for feat, imp in exp.data.items():
                        all_feature_importances[feat] = all_feature_importances.get(feat, 0.0) + imp

        # Normalize feature importances by number of instances
        num_instances = len(instances)
        averaged_importances = {feat: imp / num_instances for feat, imp in all_feature_importances.items()}

        # Build global explanation objects
        global_rule_exp = RuleBasedExplanation("LORE", all_rules)
        global_feature_exp = FeatureImportanceExplanation("LORE", averaged_importances, global_scope=True)

        return [global_rule_exp, global_feature_exp]

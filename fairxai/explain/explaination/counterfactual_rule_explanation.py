from fairxai.explain.explaination.generic_explanation import GenericExplanation


class CounterfactualRuleExplanation(GenericExplanation):
    """
    Counterfactual Rule Explanation.
    Shows minimal feature changes required to flip the prediction.
    """

    def __init__(self, explainer_name: str, counterfactual_rules: list[dict], explanation_type=None):
        explanation_type = explanation_type or self.LOCAL_EXPLANATION

        data = {
            "rules": counterfactual_rules,
            "rules_formatted": [self._format_rule(r) for r in counterfactual_rules]
        }

        super().__init__(explainer_name, explanation_type, data)

    def _format_rule(self, rule: dict) -> str:
        """Convert a counterfactual dict into a readable string."""
        return " AND ".join(f"{k} → {v}" for k, v in rule.items())

    def to_dict(self) -> dict:
        """Serializable representation."""
        return {
            "explainer_name": self.explainer_name,
            "explanation_type": "CounterfactualRuleExplanation",
            "rules": self.data["rules"],
            "rules_formatted": self.data["rules_formatted"]
        }

    def visualize(self):
        """Return a structure ready for Streamlit."""
        return self.to_dict()

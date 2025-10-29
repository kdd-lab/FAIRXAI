from fairxai.explain.explaination.generic_explanation import GenericExplanation


class CounterfactualRuleExplanation(GenericExplanation):
    """
    Counterfactual Rule Explanation.

    Shows small modifications to input features that would change the model's prediction.
    Provides insight into "what-if" scenarios.

    Example data:
        [
            {"income": "increase by 5000", "loan_approved": "True"},
            {"age": "decrease by 5", "loan_approved": "True"}
        ]

    Visualization:
        Displayed as a compact text summary of conditions and outcomes.
    """

    def __init__(self, explainer_name: str, counterfactual_rules: list[dict]):
        data = {"counterfactual_rules": counterfactual_rules}
        super().__init__(explainer_name, self.LOCAL_EXPLANATION, data)

    def visualize(self):
        """
        Display counterfactual rules in a readable format.
        """
        print(f"\n[Counterfactual Rule Explanation - {self.explainer_name}]")
        for idx, rule in enumerate(self.data.get("counterfactual_rules", []), 1):
            conditions = " AND ".join(f"{k} → {v}" for k, v in rule.items())
            print(f"{idx}. {conditions}")

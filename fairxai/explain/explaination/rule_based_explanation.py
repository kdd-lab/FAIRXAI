from fairxai.explain.explaination.generic_explanation import GenericExplanation


class RuleBasedExplanation(GenericExplanation):
    """
    Rule-Based Explanation.

    Provides a set of symbolic "if–then" rules explaining model behavior.
    Usually a global explanation representing the overall model logic.

    Example data:
        [
            "IF income > 50000 AND age < 40 THEN class = 'high spender'",
            "IF education = 'PhD' THEN class = 'premium customer'"
        ]

    Visualization:
        Displayed as a numbered list of rules for clarity."""

    def __init__(self, explainer_name: str, rules: list[str]):
        data = {"rules": rules}
        super().__init__(explainer_name, self.GLOBAL_EXPLANATION, data)

    def visualize(self):
        """
        Print the set of rules in a formatted, human-readable way.
        """
        print(f"\n[Rule-Based Explanation - {self.explainer_name}]")
        for i, rule in enumerate(self.data.get("rules", []), 1):
            print(f"{i}. {rule}")

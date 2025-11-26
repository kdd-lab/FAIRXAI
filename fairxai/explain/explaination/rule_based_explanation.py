from fairxai.explain.explaination.generic_explanation import GenericExplanation


class RuleBasedExplanation(GenericExplanation):
    """
    Rule-Based Explanation.

    Handles symbolic "if–then" rules. Rules should be passed in structured form,
    and the explanation can be serialized and consumed by Streamlit.

    Example data:
    [
        "IF income > 50000 AND age < 40 THEN class = 'high spender'",
        "IF education = 'PhD' THEN class = 'premium customer'"
    ]
    """

    def __init__(self, explainer_name: str, rules: list[dict], explanation_type=None):
        # rules: list of structured rule dicts
        explanation_type = explanation_type or self.GLOBAL_EXPLANATION

        data = {
            "rules": rules,
            "rules_formatted": [self._format_rule(r) for r in rules]
        }

        super().__init__(explainer_name, explanation_type, data)

    def _format_rule(self, rule: dict) -> str:
        """Convert a structured rule to a readable string."""
        conds = rule.get("conditions", [])
        outcome = rule.get("outcome", {})

        cond_str = " AND ".join(
            f"{c['feature']} {c['operator']} {c['value']}"
            for c in conds
        )

        out_str = " AND ".join(
            f"{k} = '{v}'" for k, v in outcome.items()
        )

        return f"IF {cond_str} THEN {out_str}"

    def to_dict(self) -> dict:
        """Return fully serializable representation (for JSON or Streamlit)."""
        return {
            "explainer_name": self.explainer_name,
            "explanation_type": "RuleBasedExplanation",
            "rules": self.data["rules"],
            "rules_formatted": self.data["rules_formatted"]
        }

    def visualize(self):
        """
        Return the formatted rules rather than printing them.
        Suitable for Streamlit.
        """
        return self.to_dict()

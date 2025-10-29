from tabulate2 import tabulate

from fairxai.explain.explaination.generic_explanation import GenericExplanation


class CounterExampleExplanation(GenericExplanation):
    """
    Counter-Example Explanation.

    Provides concrete data points that are similar to the instance but with a different prediction.
    Useful for understanding the minimal changes needed for a different output.

    Example data:
        [
            {"age": 45, "income": 30000, "prediction": "Rejected"},
            {"age": 45, "income": 35000, "prediction": "Approved"}
        ]

    Visualization:
        Displayed in tabular format for readability (CLI or GUI).
    """

    def __init__(self, explainer_name: str, counter_examples: list[dict]):
        data = {"counter_examples": counter_examples}
        super().__init__(explainer_name, self.LOCAL_EXPLANATION, data)

    def visualize(self):
        """
        Display counter-examples in a tabular format.
        """

        examples = self.data.get("counter_examples", [])
        if not examples:
            print("[INFO] No counter examples available.")
            return

        print(f"\n[Counter Example Explanation - {self.explainer_name}]")
        print(tabulate(examples, headers="keys", tablefmt="pretty"))

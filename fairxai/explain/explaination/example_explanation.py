from tabulate2 import tabulate

from fairxai.explain.explaination.generic_explanation import GenericExplanation


class ExampleExplanation(GenericExplanation):
    """
    Example Explanation.

    Shows representative or similar instances to the target instance.
    Often used in example-based explainers (KNN, ProtoDash, etc.).

    Example data:
        [
            {"id": 101, "similarity": 0.95, "label": "Approved"},
            {"id": 107, "similarity": 0.90, "label": "Approved"}
        ]

    Visualization:
        Displayed as a table showing the closest or most similar examples.
    """

    def __init__(self, explainer_name: str, examples: list[dict]):
        data = {"examples": examples}
        super().__init__(explainer_name, self.LOCAL_EXPLANATION, data)

    def visualize(self):
        """
        Display representative examples as a simple table.
        """
        examples = self.data.get("examples", [])
        if not examples:
            print("[INFO] No examples available.")
            return

        print(f"\n[Example Explanation - {self.explainer_name}]")
        print(tabulate(examples, headers="keys", tablefmt="grid"))

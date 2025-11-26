class GenericExplanation:
    """
    Base class for all explanations in the framework.

    It provides a consistent structure for:
    - storing explanation metadata,
    - storing explanation payloads,
    - returning serializable dictionaries,
    - providing Streamlit-friendly visualization output.

    Subclasses must override `visualize()` to return a structure that can be
    rendered by Streamlit (NOT printed) and may optionally extend `to_dict()`.
    """

    # Explanation type constants
    LOCAL_EXPLANATION = "local"
    GLOBAL_EXPLANATION = "global"

    def __init__(self, explainer_name: str, explanation_type: str, data: dict):
        """
        Initialize a generic explanation object.

        Parameters:
            explainer_name (str): Name of the explainer (e.g., SHAP, LIME).
            explanation_type (str): One of local/global.
            data (dict): Explanation payload (already structured and serializable).
        """
        self.explainer_name = explainer_name
        self.explanation_type = explanation_type
        self.data = data  # subclasses decide the structure of this dict

    def to_dict(self) -> dict:
        """
        Return a fully serializable representation of the explanation.

        Every explanation—rule-based, counterfactual, feature-importance, etc.—
        will produce a JSON-ready dictionary with a consistent schema.
        """
        return {
            "explainer_name": self.explainer_name,
            "explanation_type": self.explanation_type,
            "payload": self.data
        }

    def visualize(self):
        """
        Subclasses must override this and return a Streamlit-friendly structure.

        visualize() MUST:
        - NOT print anything
        - NOT generate plots directly
        - return a dict/list/string that Streamlit can render

        This ensures the visualization responsibility stays with the frontend,
        not the explanation object.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.visualize() must be implemented by subclasses."
        )

from fairxai.explain.explaination.generic_explanation import GenericExplanation


class FeatureImportanceExplanation(GenericExplanation):
    """
    Serializable feature importance explanation.
    """

    def __init__(self, explainer_name: str, data: dict, global_scope: bool = False):
        explanation_type = (
            self.GLOBAL_EXPLANATION if global_scope else self.LOCAL_EXPLANATION
        )

        # sort once and store
        sorted_features, sorted_importances = self._sort(data)

        payload = {
            "raw_importances": data,
            "sorted_features": sorted_features,
            "sorted_importances": sorted_importances
        }

        super().__init__(explainer_name, explanation_type, payload)

    def _sort(self, data: dict):
        items = sorted(data.items(), key=lambda x: x[1], reverse=True)
        return [k for k, v in items], [v for k, v in items]

    def to_dict(self) -> dict:
        """Return a JSON-serializable structure."""
        return {
            "explainer_name": self.explainer_name,
            "explanation_type": "FeatureImportanceExplanation",
            "sorted_features": self.data["sorted_features"],
            "sorted_importances": self.data["sorted_importances"],
            "raw_importances": self.data["raw_importances"]
        }

    def visualize(self):
        """
        Return data for Streamlit, not a plot.
        Streamlit can render using st.bar_chart or st.dataframe.
        """
        return self.to_dict()

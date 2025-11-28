from typing import Any, Dict, List, Optional, Tuple

from fairxai.explain.explaination.generic_explanation import GenericExplanation


class FeatureImportanceExplanation(GenericExplanation):
    """
    Serializable feature importance explanation with optional visualization.

    :param explainer_name: name of the explainer that produced this explanation
    :param data: mapping from feature identifier to numeric importance (e.g. "i,j" -> float)
    :param visualization: optional dict containing visual assets (base64 PNGs, shape, metadata)
    :param global_scope: whether the explanation is global
    """

    def __init__(
        self,
        explainer_name: str,
        data: Dict[str, float],
        visualization: Optional[Dict[str, Any]] = None,
        global_scope: bool = False,
    ):
        explanation_type = self.GLOBAL_EXPLANATION if global_scope else self.LOCAL_EXPLANATION

        # sort only numeric items and store sorted lists
        sorted_features, sorted_importances = self._sort(data)

        payload = {
            "raw_importances": data,
            "sorted_features": sorted_features,
            "sorted_importances": sorted_importances,
        }

        # attach visualization optionally, but do not include it in the ordering logic
        if visualization is not None:
            payload["visualization"] = visualization

        super().__init__(explainer_name, explanation_type, payload)

    def _sort(self, data: Dict[str, Any]) -> Tuple[List[str], List[float]]:
        """
        Sort only numeric entries in `data`. Non-numeric keys are ignored for sorting.
        Returns (sorted_feature_keys, sorted_importances).
        """
        numeric_items = []
        for k, v in data.items():
            try:
                numeric_value = float(v)
            except (TypeError, ValueError):
                continue
            numeric_items.append((k, numeric_value))

        numeric_items.sort(key=lambda x: x[1], reverse=True)
        return [k for k, _ in numeric_items], [v for _, v in numeric_items]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable structure including optional visualization."""
        out = {
            "explainer_name": self.explainer_name,
            "explanation_type": "FeatureImportanceExplanation",
            "sorted_features": self.data["sorted_features"],
            "sorted_importances": self.data["sorted_importances"],
            "raw_importances": self.data["raw_importances"],
        }
        if "visualization" in self.data:
            out["visualization"] = self.data["visualization"]
        return out

    def visualize(self) -> Dict[str, Any]:
        """
        Return data for UI libraries (e.g. Streamlit). This includes visualization if present.
        """
        return self.to_dict()

import matplotlib.pyplot as plt

from fairxai.explain.explaination.generic_explanation import GenericExplanation


class FeatureImportanceExplanation(GenericExplanation):
    """
    Explanation subclass representing feature importance values.
    Can be used for both local and global explanations.

    Purpose:
        Represents how much each feature contributes to the prediction.
        Can be local (for a single case) or global (for the entire model).

    Typical examples: LIME, SHAP, Permutation Importance.

    Data format:
        {
            "age": 0.4,
            "income": 0.2,
            "education": 0.1
        }

    Visualization:
        A bar chart sorted by importance.
        Easy to understand, widely used in explainability dashboards.
    """

    def __init__(self, explainer_name: str, data: dict, global_scope: bool = False):
        """
        Initialize a FeatureImportanceExplanation instance.

        Parameters:
            explainer_name (str): Name of the explainer (e.g., "LIME", "SHAP", "PermutationImportance").
            data (dict): Dictionary mapping feature names to their importance values.
            global_scope (bool): If True, this is a global explanation (entire model),
                                 if False, this is a local explanation (single prediction).
        """
        explanation_type = (
            self.GLOBAL_EXPLANATION if global_scope else self.LOCAL_EXPLANATION
        )
        super().__init__(explainer_name, explanation_type, data)

    def visualize(self):
        """
        Visualize the feature importance values as a sorted bar chart.

        The visualization process:
        1. Sort features by their importance values (descending order)
        2. Create a horizontal bar chart with sorted features

        This method orchestrates the visualization workflow by delegating
        to specialized private methods for better code organization.
        """
        features, importances = self._sort_features_by_importance()
        self._create_bar_chart(features, importances)

    def _sort_features_by_importance(self) -> tuple[list, list]:
        """
        Sort features by their importance values in descending order.

        This method extracts feature names and importance values from the data dictionary,
        then sorts them so that the most important features appear first. This is useful
        for creating more readable visualizations where the most relevant features are
        displayed prominently.

        Returns:
            tuple[list, list]: A tuple containing:
                - sorted_features: List of feature names sorted by importance (highest first)
                - sorted_importances: List of importance values in corresponding order

        Example:
            Input data: {"age": 0.2, "income": 0.4, "education": 0.1}
            Returns: (["income", "age", "education"], [0.4, 0.2, 0.1])
        """
        # Extract feature names and importance values from the data dictionary
        features = list(self.data.keys())
        importances = list(self.data.values())

        # Create indices sorted by importance values in descending order
        sorted_indices = sorted(
            range(len(importances)),
            key=lambda i: importances[i],
            reverse=True
        )

        # Reorder features and importances based on sorted indices
        sorted_features = [features[i] for i in sorted_indices]
        sorted_importances = [importances[i] for i in sorted_indices]

        return sorted_features, sorted_importances

    def _create_bar_chart(self, features: list, importances: list) -> None:
        """
        Create and display a horizontal bar chart for feature importances.

        This method generates a matplotlib horizontal bar chart where:
        - Y-axis shows feature names (inverted so the most important is at the top)
        - X-axis shows importance values
        - Chart includes title with explainer name for context

        The horizontal bar chart format is preferred for feature importance
        because it handles long feature names better than vertical bars and
        is easier to read when there are many features.

        Parameters:
            features (list): List of feature names (should be pre-sorted by importance).
            importances (list): List of importance values corresponding to features.

        Note:
            This method displays the chart immediately. For dashboard integration,
            consider extending this to return the figure object or save to file.
        """
        # Create a new figure with the specified size (width=8, height=5 inches)
        plt.figure(figsize=(8, 5))

        # Create a horizontal bar chart
        plt.barh(features, importances)

        # Invert y-axis so the most important feature appears at the top
        plt.gca().invert_yaxis()

        # Add title including the explainer name for context
        plt.title(f"Feature Importance - {self.explainer_name}")

        # Label the x-axis to clarify what the values represent
        plt.xlabel("Importance")

        # Adjust the layout to prevent label cutoff
        plt.tight_layout()

        # Display the chart
        plt.show()

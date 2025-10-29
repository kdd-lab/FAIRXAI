from typing import List

from fairxai.explain.explainer_manager.explainer_manager import ExplainerManager


class GuidedExplanationWizard:
    """
    Interactive wizard that guides the user step-by-step through selecting:
      1. Dataset
      2. Model / Black Box
      3. Explainer (filtered by compatibility)
      4. Visualization mode

    The wizard dynamically filters only compatible options at each step.
    It is interface-agnostic and can use CLI or GUI handlers for user interaction.
    """

    def __init__(self, io_handler):
        """
        Represents a class responsible for managing explanations and handling input/output operations.

        Attributes:
        manager (ExplainerManager): An instance of ExplainerManager for handling explanation-related tasks.
        io (Any): Input/output handler used for processing input/output operations.
        state (dict): Maintains the state information related to the class usage and its processing.

        """
        self.manager = ExplainerManager()
        self.io = io_handler
        self.state = {}

    # ======================================================================
    # MAIN ENTRY POINT
    # ======================================================================
    def run(self, datasets_available: List[str], models_available: List[str]) -> dict:
        """
        Runs the guided selection process sequentially.

        Args:
            datasets_available: List of available dataset names or types.
            models_available: List of available model names or types.

        Returns:
            dict: Final configuration with dataset, model, explainer, and visualization.
        """
        print("\n[INFO] Starting guided explanation wizard...")

        self._select_dataset(datasets_available)
        self._select_model(models_available)
        self._select_explainer()
        self._select_visualization()

        print("\n[INFO] Wizard completed successfully!")
        print(f"[INFO] Final configuration: {self.state}")
        return self.state

    # ======================================================================
    # STEP 1 - Dataset selection
    # ======================================================================
    def _select_dataset(self, datasets):
        """Prompt user to select a dataset."""
        self.state['dataset'] = self.io.ask_choice(
            title="Step 1: Select Dataset",
            options=datasets,
            message="Choose a dataset:"
        )

    # ======================================================================
    # STEP 2 - Model selection
    # ======================================================================
    def _select_model(self, models):
        """Prompt user to select a model (filtering can be applied if needed)."""
        # If you later add dataset-model compatibility filtering, it can go here.
        self.state['model'] = self.io.ask_choice(
            title="Step 2: Select Model / Black Box",
            options=models,
            message="Choose a model:"
        )

    # ======================================================================
    # STEP 3 - Explainer selection
    # ======================================================================
    def _select_explainer(self):
        """Prompt user to select an explainer compatible with chosen dataset and model."""
        dataset_type = self._extract_type(self.state['dataset'])
        model_type = self._extract_type(self.state['model'])

        explainers = self.manager.list_compatible_explainers(dataset_type, model_type)

        if not explainers:
            raise ValueError(
                f"[ERROR] No compatible explainer found for dataset '{dataset_type}' and model '{model_type}'."
            )

        self.state['explainer'] = self.io.ask_choice(
            title="Step 3: Select Explainer",
            options=explainers,
            message="Choose a compatible explainer:"
        )

    # ======================================================================
    # STEP 4 - Visualization selection
    # ======================================================================
    def _select_visualization(self):
        """Prompt user to select a visualization mode supported by the selected explainer."""
        explainer_name = self.state['explainer']
        explainer_cls = self.manager.explainers[explainer_name]

        # Check if the explainer class provides a method to list visualization modes
        if hasattr(explainer_cls, "list_visualization_modes"):
            viz_modes = explainer_cls.list_visualization_modes()
        else:
            raise ValueError("No visualization modes defined for this explainer.")

        self.state['visualization'] = self.io.ask_choice(
            title="Step 4: Select Visualization Mode",
            options=viz_modes,
            message="Choose a visualization mode:"
        )

    # ======================================================================
    # UTILITY
    # ======================================================================
    def _extract_type(self, name_or_obj: str) -> str:
        """
        Extracts a simplified 'type' string from a dataset or model name/object.

        Example:
            "Adult Income Dataset" → "adult_income"
            "RandomForestClassifier" → "random_forest"
        """
        if isinstance(name_or_obj, str):
            return name_or_obj.lower().replace(" ", "_")
        return type(name_or_obj).__name__.lower()

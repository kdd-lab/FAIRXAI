import os
from typing import Optional, Any, Type

import joblib

from fairxai.bbox import AbstractBBox
from fairxai.logger import logger


class SklearnBBox(AbstractBBox):
    """
    Wrapper for scikit-learn models to unify interface with AbstractBBox.
    This adapter handles trained models, saving/loading, and prediction methods.
    """

    def __init__(
            self,
            model: Optional[Any] = None,
            model_type: str = "sklearn_generic",
            model_name: str = "sklearn_model"
    ):
        """
        Initialize the wrapper.

        Args:
            model: Optional scikit-learn model instance.
            model_type: Logical model type (e.g., 'sklearn_random_forest').
            model_name: Optional name for tracking.
        """
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.framework = "sklearn"
        self.model_name = model_name

        if self.model is not None:
            logger.info(f"SklearnBBox wrapping existing model '{model_name}'")
        else:
            logger.warning(f"SklearnBBox initialized without a model instance. Load or assign model before use.")

    # ---------------------------
    # Prediction methods
    # ---------------------------
    def predict(self, X):
        """Predict labels for given input."""
        if self.model is None:
            raise ValueError("No model instance available. Cannot predict.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict probabilities for given input, if available."""
        if self.model is None:
            raise ValueError("No model instance available. Cannot predict_proba.")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("This scikit-learn model does not support predict_proba.")

    # ---------------------------
    # Persistence methods
    # ---------------------------
    def save(self, path: str):
        """
        Save the entire sklearn model to disk.

        Args:
            path: File path to save the model.
        """
        if self.model is None:
            raise ValueError("No model instance to save.")
        joblib.dump(self.model, path)
        logger.info(f"SklearnBBox model '{self.model_name}' saved to {path}")

    def load(self, path: str, expected_cls: Optional[Type] = None):
        """
        Load the sklearn model from disk and optionally validate its type.

        Args:
            path: Path to the .pkl or .joblib file.
            expected_cls: Optional expected model class (for validation).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file '{path}' does not exist.")

        loaded_model = joblib.load(path)

        # Optional: safety check
        if expected_cls is not None and not isinstance(loaded_model, expected_cls):
            raise TypeError(
                f"Loaded model type '{type(loaded_model).__name__}' does not match expected type '{expected_cls.__name__}'."
            )

        self.model = loaded_model
        logger.info(f"SklearnBBox model '{self.model_name}' loaded from {path} "
                    f"({type(self.model).__name__})")

import os
from typing import Optional, Any

import joblib

from fairxai.bbox import AbstractBBox
from fairxai.logger import logger


# =============================================
# SklearnBBox: Wrapper for scikit-learn models
# =============================================
class SklearnBBox(AbstractBBox):
    """
    Adapter for scikit-learn-like models.

    This wrapper allows any scikit-learn classifier/regressor
    to be used as an AbstractBBox in the framework.
    It supports pre-trained models, saving/loading from file,
    and provides predict/predict_proba methods.
    """

    def __init__(self, model: Optional[Any] = None, model_type: str = "sklearn_generic",
                 model_name: str = "sklearn_model"):
        """
        Initialize the wrapper.

        Args:
            model: Optional scikit-learn trained model instance.
            model_type: Logical type of the model for compatibility checks.
            model_name: Name of the model (used for tracking/registry).
        """
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.model_name = model_name
        self.framework = "sklearn"

        if model is not None:
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
        Save the model to disk using joblib.

        Args:
            path: File path to save the model.
        """
        if self.model is None:
            raise ValueError("No model instance to save.")
        joblib.dump(self.model, path)
        logger.info(f"SklearnBBox model '{self.model_name}' saved to {path}")

    def load(self, path: str):
        """
        Load a model from disk using joblib.

        Args:
            path: File path to load the model from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file '{path}' does not exist.")
        self.model = joblib.load(path)
        logger.info(f"SklearnBBox model '{self.model_name}' loaded from {path}")

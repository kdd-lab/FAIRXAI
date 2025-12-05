import os
from typing import Optional, Any, Type, Dict

import torch
import torch.nn as nn

from fairxai.bbox import AbstractBBox
from fairxai.logger import logger


class TorchBBox(AbstractBBox):
    """
    Wrapper class for PyTorch models, providing a unified interface consistent
    with AbstractBBox and SklearnBBox.

    Supports:
    - Full-model loading (torch.save(model, path)).
    - State-dict loading (torch.save(model.state_dict(), path)).
    - Auto-handling of DataParallel prefixes ("module.").
    - Safe device mapping (CPU/GPU).
    - Saving structured payloads containing:
        { "state_dict", "init_args", "meta", ... }

    The design allows dynamic model reconstruction without needing an instantiated
    model at wrapper creation time.
    """

    def __init__(
            self,
            model: Optional[nn.Module] = None,
            device: str = "cpu",
            model_name: str = "torch_model"
    ):
        """
        Initialize the TorchBBox.

        Args:
            model: Optional PyTorch model instance.
            device: Device to place the model on ('cpu' or 'cuda').
            model_name: Logical model identifier.
        """
        super().__init__()
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.framework = "torch"
        self.model_name = model_name

        if self.model is not None:
            self.model.to(self.device)
            logger.info(f"TorchBBox wrapping model '{model_name}' on device {self.device}")
        else:
            logger.warning("TorchBBox initialized without a model instance. Use load() before prediction.")

    # -------------------------------------------------------------------------
    # Prediction API
    # -------------------------------------------------------------------------
    def predict(self, X: Any) -> torch.Tensor:
        """
        Run forward inference.

        Args:
            X: Input tensor or data convertible to torch.Tensor.

        Returns:
            Output tensor.

        Raises:
            ValueError: If the model is not loaded.
        """
        if self.model is None:
            raise ValueError("Cannot predict: no model instance loaded.")

        self.model.eval()
        with torch.no_grad():
            # Convert non-tensor input (e.g., numpy array) to tensor
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X)

            X = X.to(self.device)
            return self.model(X)

    def predict_proba(self, X: Any) -> torch.Tensor:
        """
        Predict class probabilities for the given input.

        This method assumes that the model outputs either:
        - raw logits (shape: [N, C]) → softmax will be applied
        - probabilities already (values in [0, 1] summing to 1)

        Args:
            X: Input data as a torch.Tensor or convertible to one.

        Returns:
            A tensor of probabilities with shape [N, C].

        Raises:
            ValueError: If model is not loaded or outputs invalid probability format.
        """
        if self.model is None:
            raise ValueError("Cannot predict_proba: no model instance loaded.")

        self.model.eval()
        with torch.no_grad():

            # Convert input to tensor if needed
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X)

            X = X.to(self.device)

            # Forward pass
            out = self.model(X)

            # Validate output tensor
            if not isinstance(out, torch.Tensor):
                raise ValueError(
                    "Model output is not a tensor. Cannot compute probabilities."
                )

            # Case 1: Model outputs logits → apply softmax
            if out.dim() == 2:
                # Softmax along class dimension
                probs = torch.softmax(out, dim=1)
                return probs

            # Case 2: Binary model outputting shape [N] or [N,1]
            if out.dim() == 1:
                # Apply sigmoid for binary classification
                probs_pos = torch.sigmoid(out)
                probs = torch.stack([1 - probs_pos, probs_pos], dim=1)
                return probs

            if out.dim() == 2 and out.shape[1] == 1:
                probs_pos = torch.sigmoid(out.squeeze(1))
                probs = torch.stack([1 - probs_pos, probs_pos], dim=1)
                return probs

            raise ValueError(
                f"Unexpected model output shape for predict_proba: {out.shape}. "
                "Expected [N], [N,1], or [N,C] logits."
            )

    # -------------------------------------------------------------------------
    # Saving API
    # -------------------------------------------------------------------------
    def save(
            self,
            path: str,
            save_state_dict: bool = True,
            init_args: Optional[Dict[str, Any]] = None,
            extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save the model to disk.

        Two modes:
        - save_state_dict=True: saves a structured dict containing state_dict,
          init_args, and metadata (recommended).
        - save_state_dict=False: saves the full PyTorch model object
          (less portable, not recommended for long-term storage).

        Args:
            path: Output file path (.pt or .pth).
            save_state_dict: Whether to save only weights + metadata.
            init_args: Model initialization arguments (important for reloading).
            extra: Additional metadata to store with the model.

        Raises:
            ValueError: If model instance is missing.
        """
        if self.model is None:
            raise ValueError("Cannot save: no model instance available.")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if save_state_dict:
            payload = {
                "state_dict": self.model.state_dict(),
                "init_args": init_args or {},
                "meta": {
                    "model_name": self.model_name,
                    "framework": "torch",
                    "torch_version": torch.__version__
                }
            }
            if extra:
                payload["extra"] = extra

            torch.save(payload, path)
            logger.info(f"TorchBBox saved state_dict for '{self.model_name}' to {path}")
        else:
            # Save full model object
            torch.save(self.model, path)
            logger.info(f"TorchBBox saved full model object for '{self.model_name}' to {path}")

    # -------------------------------------------------------------------------
    # Utility: remove "module." prefix when model was saved with DataParallel
    # -------------------------------------------------------------------------
    def _strip_dataparallel(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove 'module.' prefix from state_dict keys if model was wrapped
        in DataParallel during training.

        Args:
            state_dict: Original state dictionary.

        Returns:
            Cleaned state dictionary.
        """
        new_sd = {}
        for k, v in state_dict.items():
            # If the model was trained with nn.DataParallel
            if k.startswith("module."):
                new_sd[k[len("module."):]] = v
            else:
                new_sd[k] = v
        return new_sd

    # -------------------------------------------------------------------------
    # Loading API
    # -------------------------------------------------------------------------
    def load(
            self,
            path: str,
            model_class: Optional[Type[nn.Module]] = None,
            init_args: Optional[Dict[str, Any]] = None,
            strict: bool = True,
            map_location: Optional[str] = None
    ) -> None:
        """
        Load a PyTorch model from disk.

        Supports:
        - Full model objects (torch.save(model)).
        - State dictionaries (torch.save(model.state_dict())).
        - Structured payloads:
            {"state_dict": ..., "init_args": ..., "meta": ...}

        Args:
            path: Path to .pt/.pth file.
            model_class: Required when file contains only a state_dict.
            init_args: Constructor arguments for model_class.
            strict: Whether to enforce exact state_dict matching.
            map_location: Optional device remapping during load.

        Raises:
            FileNotFoundError: If file does not exist.
            TypeError: If state_dict is loaded but no model_class is provided.
            RuntimeError: If loading fails due to architectural mismatch.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file '{path}' does not exist.")

        # Default map_location: follow wrapper device
        map_loc = map_location or ("cpu" if str(self.device).startswith("cpu") else self.device)

        try:
            loaded = torch.load(path, map_location=map_loc, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Error loading '{path}': {e}")

        # --------------------------------------------------------------
        # Case 1: File contains a complete nn.Module
        # --------------------------------------------------------------
        if isinstance(loaded, nn.Module):
            self.model = loaded.to(self.device)
            logger.info(f"TorchBBox loaded full nn.Module from {path}")
            # for name, module in self.model.named_modules():
            #     print(f"{name} -> {module}")
            return

        # --------------------------------------------------------------
        # Case 2: File contains dict — may be a state_dict or payload
        # --------------------------------------------------------------
        if isinstance(loaded, dict):

            # Case 2a — structured payload saved by save()
            if "state_dict" in loaded:
                state_dict = loaded["state_dict"]
                payload_init_args = loaded.get("init_args", {})
                init_args = init_args or payload_init_args
            else:
                # Case 2b — raw state_dict
                state_dict = loaded

            # If model instance already exists, load into it
            if self.model is not None:
                try:
                    sd = self._strip_dataparallel(state_dict)
                    self.model.load_state_dict(sd, strict=strict)
                    self.model.to(self.device)
                    logger.info(f"TorchBBox loaded state_dict into existing model instance")
                    return
                except Exception as e:
                    logger.warning(f"Failed loading into existing model: {e}. Attempting new instance.")

            # If no model_class provided → cannot reconstruct architecture
            if model_class is None:
                raise TypeError(
                    "The file contains only a state_dict. "
                    "You must provide `model_class` and optional `init_args`."
                )

            # Instantiate the model class
            try:
                model_instance = model_class(**(init_args or {}))
            except Exception as e:
                raise RuntimeError(f"Failed to instantiate model_class {model_class}: {e}")

            # Load weights
            try:
                sd = self._strip_dataparallel(state_dict)
                model_instance.load_state_dict(sd, strict=strict)
            except RuntimeError as e:
                raise RuntimeError(
                    f"State_dict loading failed: {e}. "
                    "Check model_class and init_args."
                )

            self.model = model_instance.to(self.device)
            logger.info(f"TorchBBox instantiated model_class and loaded state_dict successfully")
            return

        # --------------------------------------------------------------
        # Case 3 — unsupported file format
        # --------------------------------------------------------------
        raise TypeError(
            f"Unrecognized PyTorch file format (type={type(loaded)}). "
            "Expected nn.Module or dict containing state_dict."
        )

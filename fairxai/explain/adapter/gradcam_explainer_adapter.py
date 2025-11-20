from typing import Any, Optional, Dict, cast, List

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from fairxai.bbox import AbstractBBox
from fairxai.explain.adapter.generic_explainer_adapter import GenericExplainerAdapter


class GradCamExplainerAdapter(GenericExplainerAdapter):
    """
    Adapter for Grad-CAM explanations using the pytorch-grad-cam library.
    Only supports image datasets and models with spatial activations (CNNs/ViTs).
    """
    explainer_name = "gradcam"
    supported_datasets = ["image"]
    # We enforce the usage of TorchBBox (PyTorch models) for spatial activations.
    supported_models = ["torch_cnn", "torch_mlp", "torch_rnn", "torch_gru", "torch_generic", "torch_transformer"]

    def __init__(self, model: AbstractBBox, dataset):
        """
        Initialize adapter with wrapped model and dataset instance.
        Args:
            model: a BBox wrapper whose .model is a torch.nn.Module
            dataset: dataset instance for image explanation
        """
        super().__init__(model, dataset)
        self.torch_model = self._extract_torch_module(model)
        self.device = next(self.torch_model.parameters()).device if hasattr(self.torch_model,
                                                                            "parameters") else torch.device("cpu")
        # User may specify the target layer name in dataset metadata or via parameters; we allow override in instance_params.
        self.target_layer = None

    def _extract_torch_module(self, bbox_model: AbstractBBox) -> torch.nn.Module:
        """
        Extracts the underlying torch.nn.Module from the BBox wrapper.
        Raises if not available or not a PyTorch Module.
        """
        if not hasattr(bbox_model, "model"):
            raise RuntimeError("Provided model wrapper does not expose `.model` attribute required for Grad-CAM.")
        module = getattr(bbox_model, "model")
        if not isinstance(module, torch.nn.Module):
            raise RuntimeError("Underlying model is not a torch.nn.Module; Grad-CAM unsupported.")
        return module

    def _select_target_layer(
            self, module: torch.nn.Module, layer_name: Optional[str] = None
    ) -> torch.nn.Module:
        """
        Selects the convolutional or spatial layer to compute CAM.
        If layer_name is provided, returns that module if found;
        otherwise, tries to automatically find the last Conv2d layer.
        """
        # Case 1: user explicitly provided layer name
        if layer_name:
            named_modules: Dict[str, torch.nn.Module] = dict(module.named_modules())
            target_layer = named_modules.get(layer_name)
            if target_layer is None:
                available = list(named_modules.keys())
                raise ValueError(
                    f"Layer name '{layer_name}' not found in model modules. "
                    f"Available layers include: {available[-10:]}"
                )
            return target_layer  # this is a torch.nn.Module

        # Case 2: automatic selection — find last Conv2d layer
        conv_layers = [m for m in module.modules() if isinstance(m, torch.nn.Conv2d)]
        if conv_layers:
            return conv_layers[-1]

        # Case 3: fallback — try known transformer-like layers
        vit_like = [
            m for m in module.modules()
            if hasattr(m, "norm") or hasattr(m, "attention")
        ]
        if vit_like:
            return vit_like[-1]

        raise RuntimeError(
            "Could not automatically find a suitable convolutional or spatial layer "
            "in model; please specify `target_layer_name` in adapter params."
        )

    def explain_instance(self, instance: Any, params: Optional[Dict[str, Any]] = None):
        """
        Compute a Grad-CAM heatmap for a single image instance.
        Args:
            instance: The image data input (numpy array HxWxC or tensor)
            params: Optional dict with keys:
                - "target_layer_name": str to select layer
                - "target_class": int specifying which class to explain
                - "use_overlay": bool whether to compute overlay (if rgb image provided)
        Returns:
            GenericExplanation object containing:
                - heatmap: 2D list of floats (normalized 0..1)
                - optionally overlay: image array as list
                - metadata: shape, target_class
        """
        if params is None:
            params = {}
        target_layer_name = params.get("target_layer_name")
        target_class = params.get("target_class")
        use_overlay = params.get("use_overlay", False)

        # Extract numpy array for the image
        img_arr = instance
        if hasattr(instance, "to_array"):
            img_arr = instance.to_array()
        img_arr = np.asarray(img_arr).astype(np.float32)
        # If HxWxC format, scale if needed to 0..1
        if img_arr.max() > 1.0:
            img_arr /= 255.0
        # Convert to CHW tensor
        if img_arr.ndim == 3:
            # assume HWC
            chw = np.transpose(img_arr, (2, 0, 1))
        elif img_arr.ndim == 2:
            # single channel
            chw = np.expand_dims(img_arr, axis=0)
        else:
            raise ValueError("Instance array has unsupported number of dimensions for image.")

        input_tensor = torch.tensor(chw, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Determine target layer
        if self.target_layer is None or target_layer_name:
            self.target_layer = self._select_target_layer(self.torch_model, layer_name=target_layer_name)
        use_cuda: bool = True if self.device.type != "cpu" else False
        # Build GradCAM object
        cam = GradCAM(model=self.torch_model, target_layers=[self.target_layer])

        # Determine the target class
        if target_class is None:
            # forward pass, pick predicted class
            self.torch_model.eval()
            with torch.no_grad():
                output = self.torch_model(input_tensor)
            target_class = int(output.argmax(dim=1).item())

        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=cast(Optional[List[ClassifierOutputTarget]], targets))[
            0, :]

        heatmap = grayscale_cam
        # Norm 0..1
        heatmap -= heatmap.min()
        max_val = heatmap.max() if heatmap.max() != 0 else 1.0
        heatmap = heatmap / max_val

        payload: Dict[str, Any] = {
            "heatmap": heatmap.tolist(),
            "shape": heatmap.shape,
            "target_class": int(target_class)
        }

        if use_overlay:
            # attempt overlay on the original image for visualization
            try:
                # show_cam_on_image expects HWC scaled 0..1
                if img_arr.ndim == 3:
                    visualization = show_cam_on_image(img_arr, heatmap, use_rgb=True)
                else:
                    # replicate channel
                    vis = np.stack([img_arr, img_arr, img_arr], axis=-1)
                    visualization = show_cam_on_image(vis, heatmap, use_rgb=True)
                payload["overlay"] = visualization.tolist()
            except Exception as e:
                payload["overlay_error"] = str(e)

        return self.build_generic_explanation(data=payload, explanation_type=self.LOCAL_EXPLANATION)

    def explain_global(self):
        """
        Grad-CAM global explanation is not generally meaningful (visualizes single images),
        so this adapter raises NotImplementedError.
        """
        raise NotImplementedError("GradCamExplainerAdapter does not support global explanations.")

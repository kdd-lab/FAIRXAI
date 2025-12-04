from typing import Any, Dict, List, Optional, Tuple
import base64
from io import BytesIO

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from fairxai.bbox import AbstractBBox
from fairxai.explain.adapter.generic_explainer_adapter import GenericExplainerAdapter
from fairxai.explain.explaination.feature_importance_explanation import FeatureImportanceExplanation


class GradCamExplainerAdapter(GenericExplainerAdapter):
    """
    Grad-CAM explainer adapter for image datasets.
    Accepts np.ndarray images HxWxC (Height × Width × Channels).
    Computes Grad-CAM on a selected target layer and returns
    FeatureImportanceExplanation with pixel importances and visualization.

    :param model: a BBox wrapper exposing `.model` (torch.nn.Module)
    :param dataset: dataset object (kept for compatibility)
    """

    explainer_name = "gradcam"
    supported_datasets = ["image"]
    supported_models = ["Conv1d", "Conv2d", "Conv3d", "Sequential", "GraphModule", "TransformerEncoderLayer"]

    def __init__(self, model: AbstractBBox, dataset: Any) -> None:
        super().__init__(model, dataset)
        self.torch_model = self._extract_torch_module(model)
        self.device = next(self.torch_model.parameters()).device if any(True for _ in self.torch_model.parameters()) else torch.device("cpu")
        self.target_layer: Optional[torch.nn.Module] = None
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

    def _extract_torch_module(self, bbox_model: AbstractBBox) -> torch.nn.Module:
        if not hasattr(bbox_model, "model"):
            raise RuntimeError("Provided model wrapper does not expose `.model` required for Grad-CAM.")
        module = getattr(bbox_model, "model")
        if not isinstance(module, torch.nn.Module):
            raise RuntimeError("Underlying model is not a torch.nn.Module; Grad-CAM unsupported.")
        return module

    def _select_target_layer(self, module: torch.nn.Module, layer_name: Optional[str] = None) -> torch.nn.Module:
        if layer_name:
            named_modules: Dict[str, torch.nn.Module] = dict(module.named_modules())
            target_layer = named_modules.get(layer_name)
            if target_layer is None:
                available = list(named_modules.keys())
                raise ValueError(f"Layer '{layer_name}' not found. Available (last 10): {available[-10:]}")
            return target_layer
        conv_layers = [m for m in module.modules() if isinstance(m, torch.nn.Conv2d)]
        if conv_layers:
            return conv_layers[-1]
        vit_like = [m for m in module.modules() if hasattr(m, "norm") or hasattr(m, "attention")]
        if vit_like:
            return vit_like[-1]
        raise RuntimeError("No suitable layer found; pass `target_layer_name`.")

    def _forward_hook(self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _backward_hook(self, module: torch.nn.Module, grad_input: Tuple, grad_output: Tuple) -> None:
        self._gradients = grad_output[0].detach()

    def _register_hooks_on_layer(self, layer: torch.nn.Module) -> None:
        layer.register_forward_hook(self._forward_hook)
        layer.register_backward_hook(self._backward_hook)

    @staticmethod
    def _ndarray_to_base64(img: np.ndarray) -> str:
        """Encode a HxWxC uint8 numpy array as base64 PNG string."""
        pil = Image.fromarray(img.astype("uint8"))
        buf = BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _compute_cam_from_captured(self) -> np.ndarray:
        """Compute Grad-CAM heatmap from captured activations and gradients."""
        if self._activations is None or self._gradients is None:
            raise RuntimeError("Activations or gradients not captured for Grad-CAM.")
        weights = self._gradients.mean(dim=(1, 2), keepdim=True)  # spatial average
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
        cam = cam - cam_min
        cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
        cam_max[cam_max == 0] = 1.0
        cam = cam / cam_max
        return cam[0, 0].cpu().numpy()

    def _np_to_nhwc_tensor(self, img: np.ndarray) -> torch.Tensor:
        """
        Convert HxWxC np.ndarray to NHWC torch tensor with batch dim.
        Assumes img already has 3 channels; adds batch dimension.
        Normalizes to [0,1].
        """
        arr = img.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        arr /= 255.0
        return torch.tensor(arr).unsqueeze(0).to(self.device)  # shape (1,H,W,C)

    def explain_instance(self, instance: np.ndarray, params: Optional[Dict[str, Any]] = None) -> List[FeatureImportanceExplanation]:
        """
        Explain a single image instance using Grad-CAM.

        :param instance: np.ndarray HxWxC
        :param params: optional dict; supported keys:
                       - "target_layer_name": str
                       - "target_class": int
        :returns: list with a single FeatureImportanceExplanation
        """
        target_layer_name = params.get("target_layer_name") if params else None
        target_class = int(params.get("target_class")) if params and "target_class" in params else None

        # select target layer if needed
        if self.target_layer is None or target_layer_name:
            self.target_layer = self._select_target_layer(self.torch_model, layer_name=target_layer_name)
        self._register_hooks_on_layer(self.target_layer)

        # convert image to NHWC tensor with batch
        input_tensor = self._np_to_nhwc_tensor(instance)

        # forward pass
        self.torch_model.zero_grad()
        input_tensor.requires_grad_(True)
        output = self.torch_model(input_tensor)  # output can be spatial (1,H,W,num_classes)

        # determine target class if not provided
        if target_class is None:
            if output.ndim == 2:  # (B, num_classes)
                target_class = int(output[0].argmax().item())
            elif output.ndim == 4:  # (B, H, W, C)
                pooled = output.mean(dim=(1, 2))  # global average pooling H,W
                target_class = int(pooled[0].argmax().item())
            else:
                raise RuntimeError(f"Unexpected output shape {output.shape}")

        # backward on target logit
        scalar = output[0, target_class]
        scalar.backward(retain_graph=False)

        # compute Grad-CAM heatmap
        heatmap = self._compute_cam_from_captured()  # HxW, 0..1

        # resize heatmap to original image size
        orig_h, orig_w = instance.shape[:2]
        heatmap_img = (heatmap * 255).astype(np.uint8)
        heatmap_pil = Image.fromarray(heatmap_img)
        heatmap_resized = np.asarray(heatmap_pil).astype(np.uint8)

        # convert to 3-channel heatmap
        #heatmap_rgb = np.stack([heatmap_resized]*3, axis=-1)

        # flatten heatmap to pixel importances
        h, w = heatmap_resized.shape
        pixel_importances = {f"{i},{j}": float(heatmap_resized[i,j]/255.0) for i in range(h) for j in range(w)}

        visualization_payload = {
            "heatmap": heatmap,
            "target_class": target_class,
            "original_size": (orig_w, orig_h),
        }

        fi_explanation = FeatureImportanceExplanation(
            explainer_name=self.explainer_name,
            data=pixel_importances,
            visualization=visualization_payload,
            global_scope=False
        )
        return [fi_explanation]

    def explain_global(self) -> List[FeatureImportanceExplanation]:
        raise NotImplementedError("GradCamExplainerAdapter does not support global explanations.")

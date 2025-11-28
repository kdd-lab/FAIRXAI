from typing import Any, Dict, List, Optional, Tuple

import base64
from io import BytesIO

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.base_cam import BaseCAM  # used for typing only if available

from fairxai.bbox import AbstractBBox
from fairxai.explain.adapter.generic_explainer_adapter import GenericExplainerAdapter
from fairxai.explain.explaination.feature_importance_explanation import FeatureImportanceExplanation


class GradCamExplainerAdapter(GenericExplainerAdapter):
    """
    Grad-CAM explainer adapter for image datasets.

    This adapter:
    - Accepts Pillow Images as instances.
    - Respects a provided `hwc_permutation` (or default channel_mode).
    - Computes Grad-CAM on a selected target layer.
    - Returns a FeatureImportanceExplanation with pixel importances and optional visualization.

    :param model: a BBox wrapper exposing `.model` (torch.nn.Module)
    :param dataset: dataset object (kept for compatibility)
    :param default_channel_mode: default permutation tuple mapping CHW -> HWC (default (1,2,0))
    """

    explainer_name = "gradcam"
    supported_datasets = ["image"]

    supported_models = [
        "Conv1d",
        "Conv2d",
        "Conv3d",
        "Sequential",
        "GraphModule",
        "TransformerEncoderLayer",
    ]

    def __init__(
        self,
        model: AbstractBBox,
        dataset: Any,
        default_channel_mode: Tuple[int, int, int] = (1, 2, 0),
    ) -> None:
        super().__init__(model, dataset)
        self.torch_model = self._extract_torch_module(model)
        # set device using model parameters if available, otherwise cpu
        self.device = (
            next(self.torch_model.parameters()).device
            if any(True for _ in self.torch_model.parameters())
            else torch.device("cpu")
        )
        self.default_channel_mode = default_channel_mode
        self.target_layer = None

        # storage for hooks
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

    # -----------------------------
    # helper: extract module
    # -----------------------------
    def _extract_torch_module(self, bbox_model: AbstractBBox) -> torch.nn.Module:
        """
        Extract underlying torch.nn.Module from the BBox wrapper. Raises if not found.
        """
        if not hasattr(bbox_model, "model"):
            raise RuntimeError("Provided model wrapper does not expose `.model` attribute required for Grad-CAM.")
        module = getattr(bbox_model, "model")
        if not isinstance(module, torch.nn.Module):
            raise RuntimeError("Underlying model is not a torch.nn.Module; Grad-CAM unsupported.")
        return module

    # -----------------------------
    # target layer selection
    # -----------------------------
    def _select_target_layer(self, module: torch.nn.Module, layer_name: Optional[str] = None) -> torch.nn.Module:
        """
        Select a target convolutional/spatial layer. If layer_name is provided, returns that; otherwise tries to find last Conv2d or a vit-like layer.
        """
        if layer_name:
            named_modules: Dict[str, torch.nn.Module] = dict(module.named_modules())
            target_layer = named_modules.get(layer_name)
            if target_layer is None:
                available = list(named_modules.keys())
                raise ValueError(f"Layer name '{layer_name}' not found. Available (last 10): {available[-10:]}")
            return target_layer

        conv_layers = [m for m in module.modules() if isinstance(m, torch.nn.Conv2d)]
        if conv_layers:
            return conv_layers[-1]

        vit_like = [m for m in module.modules() if hasattr(m, "norm") or hasattr(m, "attention")]
        if vit_like:
            return vit_like[-1]

        raise RuntimeError("Could not find suitable convolutional/spatial layer; please pass target_layer_name.")

    # -----------------------------
    # hooks (methods, not nested functions)
    # -----------------------------
    def _forward_hook(self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        """Forward hook: store activations (detach to avoid memory retention)."""
        # detach to avoid keeping computation graph
        self._activations = output.detach()

    def _backward_hook(self, module: torch.nn.Module, grad_input: Tuple, grad_output: Tuple) -> None:
        """Backward hook: store gradient of the layer outputs."""
        # grad_output[0] corresponds to gradient w.r.t. output
        self._gradients = grad_output[0].detach()

    def _register_hooks_on_layer(self, layer: torch.nn.Module) -> None:
        """Register forward/backward hooks to capture activations and gradients."""
        # remove any previous hooks by reassigning (PyTorch doesn't provide easy removal without handles, so we keep it simple)
        layer.register_forward_hook(self._forward_hook)
        layer.register_backward_hook(self._backward_hook)

    # -----------------------------
    # image / tensor helpers
    # -----------------------------
    @staticmethod
    def _pil_to_chw_tensor(img: Image.Image) -> torch.Tensor:
        """
        Convert a PIL image to a (C,H,W) torch tensor in float [0,1].
        Supports grayscale and RGB.
        """
        arr = np.asarray(img).astype("float32") / 255.0
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        # shape now H,W,C
        chw = torch.tensor(arr).permute(2, 0, 1)
        return chw

    @staticmethod
    def _ndarray_to_base64(img: np.ndarray) -> str:
        """Encode a HxWxC uint8 numpy array as base64 PNG string."""
        pil = Image.fromarray(img.astype("uint8"))
        buf = BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # -----------------------------
    # grad-cam math
    # -----------------------------
    def _compute_cam_from_captured(self) -> np.ndarray:
        """
        Compute Grad-CAM heatmap from captured activations and gradients.
        Expects self._activations shape (B, C, H, W) and self._gradients same shape.
        Returns a numpy array HxW normalized to [0,1].
        """
        if self._activations is None or self._gradients is None:
            raise RuntimeError("Activations or gradients not captured for Grad-CAM computation.")

        # weights: average gradients spatially -> shape (B, C, 1, 1)
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # B,C,1,1

        # weighted sum of activations -> (B,1,H,W)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)

        # relu
        cam = F.relu(cam)

        # normalize per image in batch
        cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
        cam = cam - cam_min
        cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
        cam_max[cam_max == 0] = 1.0
        cam = cam / cam_max

        # return first item HxW
        return cam[0, 0].cpu().numpy()

    # -----------------------------
    # main API
    # -----------------------------
    def explain_instance(self, instance: Any, params: Optional[Dict[str, Any]] = None) -> List[FeatureImportanceExplanation]:
        """
        Explain a single image instance using Grad-CAM.

        :param instance: a Pillow Image representing the input image
        :param params: optional dict; supported keys:
                       - "target_layer_name": str name of module to use as target
                       - "hwc_permutation": list/tuple of 3 ints specifying CHW->HWC permutation
                       - "target_class": int index of class to explain
        :returns: list with a single FeatureImportanceExplanation
        """
        # resolve params
        target_layer_name = None
        permutation: Tuple[int, int, int] = self.default_channel_mode
        target_class: Optional[int] = None

        if params:
            if "target_layer_name" in params:
                target_layer_name = params["target_layer_name"]
            if "hwc_permutation" in params:
                perm = tuple(params["hwc_permutation"])
                if len(perm) != 3:
                    raise ValueError("hwc_permutation must be length 3")
                permutation = perm
            if "target_class" in params:
                target_class = int(params["target_class"])

        # pick or select target layer
        if self.target_layer is None or target_layer_name:
            self.target_layer = self._select_target_layer(self.torch_model, layer_name=target_layer_name)

        # register hooks on the layer (idempotent for same layer)
        self._register_hooks_on_layer(self.target_layer)

        # convert Pillow -> CHW tensor
        if not isinstance(instance, Image.Image):
            raise TypeError("Instance must be a PIL Image")
        chw = self._pil_to_chw_tensor(instance).to(self.device)

        # reorder CHW -> HWC according to provided permutation, then add batch dim => (1,H,W,C)
        hwc = chw.permute(permutation).unsqueeze(0).to(self.device)

        # forward pass
        self.torch_model.zero_grad()
        hwc.requires_grad_(True)
        output = self.torch_model(hwc)  # assume model accepts HWC batch

        # determine class if not provided
        if target_class is None:
            # if model returns logits over classes on dim=1
            target_class = int(output.argmax(dim=1).item())

        # backward on the target logit to populate gradients
        scalar = output[0, target_class]
        scalar.backward(retain_graph=False)

        # compute heatmap
        heatmap = self._compute_cam_from_captured()  # H x W, floats 0..1

        # resize heatmap to original image size
        original_size = instance.size  # (W, H)
        heatmap_img = (heatmap * 255.0).astype("uint8")
        heatmap_pil = Image.fromarray(heatmap_img).resize(original_size, resample=Image.BILINEAR)
        heatmap_resized = np.asarray(heatmap_pil).astype("uint8")  # H,W

        # convert to 3-channel heatmap for visualization
        heatmap_rgb = np.stack([heatmap_resized] * 3, axis=-1)  # H,W,3

        # build overlay: blend original (uint8) with heatmap (uint8)
        orig_uint8 = np.asarray(instance.convert("RGB")).astype("uint8")
        # ensure same shape
        if orig_uint8.shape[:2] != heatmap_rgb.shape[:2]:
            # resize heatmap_rgb to match orig
            heatmap_rgb = np.asarray(Image.fromarray(heatmap_rgb).resize((orig_uint8.shape[1], orig_uint8.shape[0]), resample=Image.BILINEAR))

        overlay = (0.5 * orig_uint8 + 0.5 * heatmap_rgb).astype("uint8")

        # encode images as base64 (for UI)
        heatmap_b64 = self._ndarray_to_base64(heatmap_rgb)
        overlay_b64 = self._ndarray_to_base64(overlay)

        # flatten numeric heatmap into dictionary "i,j" -> float
        h, w = heatmap_resized.shape
        pixel_importances: Dict[str, float] = {
            f"{i},{j}": float(heatmap_resized[i, j] / 255.0)  # normalized 0..1
            for i in range(h)
            for j in range(w)
        }

        # build visualization payload
        visualization_payload: Dict[str, Any] = {
            "heatmap_base64": heatmap_b64,
            "overlay_base64": overlay_b64,
            "shape": (h, w),
            "target_class": int(target_class),
            "original_size": original_size,  # (W,H)
        }

        # create FeatureImportanceExplanation with visualization attached
        fi_explanation = FeatureImportanceExplanation(
            explainer_name=self.explainer_name,
            data=pixel_importances,
            visualization=visualization_payload,
            global_scope=False,
        )

        return [fi_explanation]

    def explain_global(self) -> List[FeatureImportanceExplanation]:
        """
        Grad-CAM does not provide meaningful global explanations.
        Keep API compatibility by raising.
        """
        raise NotImplementedError("GradCamExplainerAdapter does not support global explanations.")

from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from fairxai.bbox import AbstractBBox
from fairxai.explain.adapter.generic_explainer_adapter import GenericExplainerAdapter
from fairxai.explain.explaination.feature_importance_explanation import FeatureImportanceExplanation


class GradCamExplainerAdapter(GenericExplainerAdapter):
    """
    Grad-CAM explainer adapter using pytorch-grad-cam.

    Accepts images in HxWxC format (NHWC) and computes Grad-CAM for a specified
    target layer and target class.

    :param model: BBox wrapper exposing `.model` (torch.nn.Module)
    :param dataset: Dataset object (kept for compatibility)
    """

    explainer_name: str = "gradcam"
    supported_datasets: List[str] = ["image"]
    supported_models: List[str] = ["Conv1d", "Conv2d", "Conv3d", "Sequential", "GraphModule", "TransformerEncoderLayer"]

    def __init__(self, model: AbstractBBox, dataset: Any) -> None:
        super().__init__(model, dataset)
        self.torch_model: torch.nn.Module = self._extract_torch_module(model)
        self.device: torch.device = next(self.torch_model.parameters()).device
        self.target_layer: Optional[torch.nn.Module] = None

    def _extract_torch_module(self, bbox_model: AbstractBBox) -> torch.nn.Module:
        """Extract the internal torch module from a BBox wrapper."""
        if not hasattr(bbox_model, "model"):
            raise RuntimeError("Provided model wrapper does not expose `.model`")
        module = getattr(bbox_model, "model")
        if not isinstance(module, torch.nn.Module):
            raise RuntimeError("Underlying model is not a torch.nn.Module")
        return module

    def _np_to_nchw_tensor(self, img: np.ndarray) -> torch.Tensor:
        """
        Convert NHWC numpy array to NCHW torch tensor with batch dimension.

        :param img: Input image (HxWxC), float or uint8
        :return: Torch tensor (1, C, H, W) normalized to [0,1]
        """
        arr = img.astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[..., None]
        # NHWC -> NCHW
        # arr = np.transpose(arr, (2, 0, 1))
        return torch.tensor(arr).unsqueeze(0).to(self.device)

    def _to_char_list(self, arr: np.ndarray):
        """Convert uint8 HxWxC or HxW array to list[str] via np.char.mod()."""
        return np.char.mod('%s', arr).tolist()

    def explain_instance(self, instance: np.ndarray, params: Optional[Dict[str, Any]] = None) -> List[
        FeatureImportanceExplanation]:
        """
        Explain a single image instance using Grad-CAM.

        :param instance: Input image as HxWxC numpy array
        :param params: Optional dictionary with keys:
                       - "target_layer": torch.nn.Module, the layer to compute Grad-CAM
                       - "target_class": int, class index to explain
        :return: List containing a single FeatureImportanceExplanation
        """
        # Extract parameters
        target_layer = params.get("target_layer") if params else None
        target_layer_name = getattr(self.torch_model, target_layer)
        print(f"Given target layer: {target_layer}")
        target_class = int(params.get("target_class")) if params and "target_class" in params else None

        if target_layer is None:
            raise ValueError("You must provide 'target_layer' in params for Grad-CAM")

        # Convert input to NCHW tensor
        input_tensor = self._np_to_nchw_tensor(instance)

        # Determine target class if not provided
        self.torch_model.eval()
        with torch.no_grad():
            output = self.torch_model(input_tensor)
            if target_class is None:
                if output.ndim == 2:  # (B, num_classes)
                    target_class = int(output[0].argmax())
                elif output.ndim == 4:  # (B,H,W,C)
                    target_class = int(output.mean(dim=(1, 2))[0].argmax())
                else:
                    raise RuntimeError(f"Unexpected output shape {output.shape}")

        # Prepare pytorch-grad-cam
        targets = [ClassifierOutputTarget(target_class)]

        # Run CAM
        with GradCAM(model=self.torch_model, target_layers=[target_layer_name]) as cam:
            grayscale_cams = cam(input_tensor=input_tensor, targets=targets)

        # CAM in [0,1], shape activation_H × activation_W
        grayscale_cam = grayscale_cams[0, :]

        # Original size
        orig_h, orig_w = instance.shape[:2]

        # Resize CAM to original image size BEFORE any other operation
        grayscale_cam_resized = np.array(
            Image.fromarray((grayscale_cam * 255).astype(np.uint8)).resize(
                (orig_w, orig_h),
                Image.Resampling.BILINEAR
            )
        ) / 255.0  # back to [0,1]

        # Original image normalized to [0,1] float RGB
        rgb_float = instance.astype(np.float32) / 255.0

        # ============================================================
        # 1) Grayscale heatmap (3-channel)
        # ============================================================
        heatmap_gray_3ch = np.stack([
                                        (grayscale_cam_resized * 255).astype(np.uint8)
                                    ] * 3, axis=-1)

        # ============================================================
        # 2) Colored heatmap
        # ============================================================
        heatmap_color = cv2.applyColorMap(
            (grayscale_cam_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # ============================================================
        # 3) Overlay (image + heatmap)
        # ============================================================
        overlay = show_cam_on_image(rgb_float, grayscale_cam_resized, use_rgb=False)
        overlay_uint8 = (overlay * 255).clip(0, 255).astype(np.uint8)

        # Flatten heatmap into pixel importance dict
        pixel_importances = {
            f"{i},{j}": float(grayscale_cam_resized[i, j])
            for i in range(orig_h)
            for j in range(orig_w)
        }

        # -----------------------------
        # Save PNGs for debug purposes
        # -----------------------------
        # Image.fromarray((grayscale_cam_resized * 255).astype(np.uint8)).save("debug_heatmap_gray.png")
        # Image.fromarray(heatmap_gray_3ch).save("debug_heatmap_gray_3ch.png")
        # Image.fromarray(heatmap_color).save("debug_heatmap_color.png")
        # Image.fromarray(overlay_uint8).save("debug_overlay.png")

        visualization_payload = {
            "target_class": target_class,
            "original_size": (orig_w, orig_h),
            #
            # # Grayscale heatmap (HxW)
            # "heatmap_gray": self._to_char_list((grayscale_cam_resized * 255).astype(np.uint8)),
            #
            # # Grayscale heatmap in RGB (HxWx3)
            # "heatmap_gray_3ch": self._to_char_list(heatmap_gray_3ch),
            #
            # # Colored heatmap (HxWx3)
            # "heatmap_color": self._to_char_list(heatmap_color),

            # Overlay image (HxWx3)
            "overlay": self._to_char_list(overlay_uint8)
        }

        fi_explanation = FeatureImportanceExplanation(
            explainer_name=self.explainer_name,
            data=pixel_importances,
            visualization=visualization_payload,
            global_scope=False
        )
        return [fi_explanation]

    def explain_global(self) -> List[FeatureImportanceExplanation]:
        """Grad-CAM does not support global explanations."""
        raise NotImplementedError("GradCamExplainerAdapter does not support global explanations.")

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image
from fairxai.explain.explaination.generic_explanation import GenericExplanation


class ExampleExplanation(GenericExplanation):
    """
    Handles example-based explanations for both tabular and image inputs.
    """

    def __init__(self, explainer_name: str, examples: list[dict]):
        data = {"examples": [self._serialize_example(ex) for ex in examples]}
        super().__init__(explainer_name, self.LOCAL_EXPLANATION, data)

    def _serialize_image(self, value: Any) -> Dict[str, str]:
        if isinstance(value, Image.Image):
            img = value
        elif np is not None and isinstance(value, np.ndarray):
            img = Image.fromarray(value.astype("uint8"))
        elif isinstance(value, (bytes, bytearray)):
            try:
                img = Image.open(BytesIO(value))
            except Exception as e:
                raise ValueError("Provided bytes are not a valid image") from e
        elif isinstance(value, str) and Path(value).exists():
            img = Image.open(value)
        else:
            raise TypeError("Unsupported image type for serialization")

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"type": "base64", "encoding": "png", "data": b64}

    def _serialize_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in example.items():
            try:
                if isinstance(v, Image.Image) or (np is not None and isinstance(v, np.ndarray)) or isinstance(v, (bytes, bytearray)) or (isinstance(v, str) and Path(v).exists()):
                    out[k] = self._serialize_image(v)
                elif isinstance(v, dict) and v.get("type") == "base64" and "data" in v:
                    out[k] = v
                else:
                    out[k] = v
            except (TypeError, ValueError):
                out[k] = str(v)
        return out

    def to_dict(self) -> Dict[str, Any]:
        raw_examples = self.data.get("examples", [])
        serialized = [self._serialize_example(ex) for ex in raw_examples]
        return {
            "explainer_name": self.explainer_name,
            "explanation_type": "ExampleExplanation",
            "payload": {"examples": serialized}
        }

    def visualize(self) -> Dict[str, Any]:
        return self.to_dict()

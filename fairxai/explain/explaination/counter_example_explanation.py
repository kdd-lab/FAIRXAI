import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from PIL import Image
from fairxai.explain.explaination.generic_explanation import GenericExplanation


class CounterExampleExplanation(GenericExplanation):
    """
    Counter-Example Explanation that supports tabular fields and images.

    Each example is a dict. If a value is:
      - PIL.Image.Image -> encoded as base64 PNG
      - numpy.ndarray -> converted to PIL.Image then encoded (if numpy available)
      - bytes -> assumed image bytes and base64-encoded
      - str path -> file read and base64-encoded (if file exists)
      - other primitive -> left as-is

    to_dict() returns a JSON-ready wrapper with the serialized examples.
    visualize() returns the same structure (no printing).
    """

    def __init__(self, explainer_name: str, counter_examples: List[Dict[str, Any]]):
        # store raw input; we'll produce a serialized payload in to_dict()
        super().__init__(explainer_name, self.LOCAL_EXPLANATION, {"counter_examples": counter_examples})

    def _serialize_image(self, value: Any) -> Dict[str, str]:
        """
        Convert various image-like inputs to a base64-encoded PNG representation.
        Returns a dict: {"type": "base64", "encoding": "png", "data": "<base64 str>"}
        """
        # PIL Image
        if isinstance(value, Image.Image):
            img = value
        # numpy array
        elif np is not None and isinstance(value, np.ndarray):
            # handle grayscale or RGB arrays; convert using PIL
            img = Image.fromarray(value.astype("uint8"))
        # bytes (raw image bytes)
        elif isinstance(value, (bytes, bytearray)):
            try:
                img = Image.open(BytesIO(value))
            except Exception as e:
                raise ValueError("Provided bytes are not a valid image") from e
        # path-like string -> try to open file
        elif isinstance(value, str) and Path(value).exists():
            img = Image.open(value)
        else:
            raise TypeError("Unsupported image type for serialization")

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"type": "base64", "encoding": "png", "data": b64}

    def _serialize_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize a single example: convert any image-like field into the base64 wrapper,
        leave other values untouched (primitives, dicts, lists).
        """
        out: Dict[str, Any] = {}
        for k, v in example.items():
            try:
                # try to detect image-likes: PIL, numpy, bytes, or existing dict wrapper
                if isinstance(v, Image.Image) or (np is not None and isinstance(v, np.ndarray)) or isinstance(v, (bytes, bytearray)) or (isinstance(v, str) and Path(v).exists()):
                    out[k] = self._serialize_image(v)
                # if already given in the serialized form (common when reloading), keep as-is
                elif isinstance(v, dict) and v.get("type") == "base64" and "data" in v:
                    out[k] = v
                else:
                    out[k] = v
            except (TypeError, ValueError):
                # fallback: include string representation
                out[k] = str(v)
        return out

    def to_dict(self) -> Dict[str, Any]:
        """
        Build the JSON-ready representation.
        """
        raw_examples = self.data.get("counter_examples", [])
        serialized = [self._serialize_example(ex) for ex in raw_examples]
        return {
            "explainer_name": self.explainer_name,
            "explanation_type": "CounterExampleExplanation",
            "payload": {"counter_examples": serialized}
        }

    def visualize(self) -> Dict[str, Any]:
        """Return the same dict that to_dict() produces (no printing)."""
        return self.to_dict()
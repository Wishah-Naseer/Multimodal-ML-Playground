import json
import os
from dataclasses import asdict, dataclass
from typing import Tuple

@dataclass
class Detection:
    box: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    label: int
    score: float

def save_detections_json(
    image_path: str,
    detections: list[Detection],
    image_size: Tuple[int, int],
    save_path: str,
    extra: dict | None = None
) -> None:
    """
    Save detections to JSON:
    {
      "image": "bus.jpg",
      "image_size": [W, H],
      "detections": [{"box":[x1,y1,x2,y2], "label": 5, "score": 0.98}, ...],
      "meta": {...}   # optional
    }
    """
    payload = {
        "image": os.path.basename(image_path),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "detections": [asdict(d) for d in detections],
        "meta": extra or {},
    }
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


from transformers import pipeline
import torch
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List, Any, Dict, Optional, TypedDict, Literal

# ---------- Typed feature schema ----------

FeatureType = Literal["image", "string", "int", "float"]

class Feature(TypedDict):
    type: FeatureType
    value: Any

# ---------- Image helpers ----------

def _decode_b64_to_pil(b64: str) -> Image.Image:
    """Convert base64 string to PIL Image (RGB)."""
    decoded = base64.b64decode(b64)
    return Image.open(BytesIO(decoded)).convert("RGB")

def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalized torch tensor (NCHW, [0,1])."""
    arr = np.array(img)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)

# ---------- Model ----------

class MultiModalModel:
    """
    Unified model handling explicitly-typed features:
      - {"type": "image",  "value": <base64-encoded image string>}  -> object detection via Faster R-CNN
      - {"type": "string", "value": <str>}                          -> BERT "fill-mask"
      - {"type": "int",    "value": <int>}                          -> x * 2
      - {"type": "float",  "value": <float>}                        -> round(x**2, 1)
    Returns a list of dicts, preserving input order.
    """

    def __init__(self, load_vision: bool = True, load_text: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize text pipeline if requested
        self.text_pipe = None
        if load_text:
            self.text_pipe = pipeline(
                "fill-mask",
                model="bert-base-uncased",
                device=0 if self.device.type == "cuda" else -1,
            )

        # Initialize vision detector if requested
        self.detector = None
        if load_vision:
            try:
                from torchvision.models.detection import fasterrcnn_resnet50_fpn
                # from torchvision.models.detection import fasterrcnn_resnet50_fpn  # type: ignore[import-not-found]
                try:
                    # torchvision >= 0.13 style
                    self.detector = fasterrcnn_resnet50_fpn(weights="DEFAULT").eval().to(self.device)
                except TypeError:
                    # older torchvision fallback
                    self.detector = fasterrcnn_resnet50_fpn(pretrained=True).eval().to(self.device)
            except Exception as e:
                print(f"[WARN] Vision model not available: {e}")
                self.detector = None

    def get_details(self) -> Dict[str, Any]:
        return {
            "model": self.__class__.__name__,
            "description": "Unified model handling typed images, strings (BERT mask), ints, and floats.",
            "capabilities": {
                "image": bool(self.detector is not None),
                "string_fill_mask": bool(self.text_pipe is not None),
                "int": True,
                "float": True,
            },
        }

    # ----- individual predictors -----

    def _predict_image(self, b64: str) -> Dict[str, Any]:
        """Run object detection on a base64-encoded image string."""
        if self.detector is None:
            return {"error": "Vision backend unavailable"}
        try:
            img = _decode_b64_to_pil(b64)
        except Exception as e:
            return {"error": f"Invalid base64 image: {e}"}
        tensor = _pil_to_tensor(img).to(self.device)
        with torch.inference_mode():
            out = self.detector(tensor)[0]
        return {
            "boxes": out["boxes"].detach().cpu().tolist(),
            "labels": [int(x) for x in out["labels"].detach().cpu().tolist()],
            "scores": out["scores"].detach().cpu().tolist(),
        }

    def _predict_string(self, s: str) -> Any:
        """Run BERT fill-mask on text."""
        if self.text_pipe is None:
            return {"error": "Text backend unavailable"}
        try:
            return self.text_pipe(s)[0]
        except Exception as e:
            return {"error": f"Text inference failed: {e}"}

    def _predict_int(self, x: int) -> int:
        """Simple int transform: multiply by 2."""
        return x * 2

    def _predict_float(self, x: float) -> float:
        """Simple float transform: square and round."""
        return round(x ** 2, 1)

    # ----- main predict -----

    def predict(self, features: List[Feature]) -> List[Dict[str, Any]]:
        """
        Process a list of typed features. Each item must be a dict:
          {"type": "<image|string|int|float>", "value": <...>}
        """
        if not isinstance(features, list) or len(features) == 0:
            raise ValueError("features must be a non-empty list")

        outputs: List[Dict[str, Any]] = []

        for idx, feat in enumerate(features):
            if not isinstance(feat, dict) or "type" not in feat or "value" not in feat:
                outputs.append({
                    "index": idx,
                    "input_type": "unsupported",
                    "result": {"error": "Each feature must be a dict with 'type' and 'value' keys"},
                })
                continue

            ftype = feat["type"]
            val = feat["value"]

            # Strict type validation per declared ftype
            if ftype == "int":
                if isinstance(val, bool) or not isinstance(val, int):
                    outputs.append({"index": idx, "input_type": "int", "result": {"error": "value must be int"}})
                    continue
                result = self._predict_int(val)

            elif ftype == "float":
                # Accept int as float input and cast
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    result = self._predict_float(float(val))
                else:
                    outputs.append({"index": idx, "input_type": "float", "result": {"error": "value must be float"}})
                    continue

            elif ftype == "string":
                if not isinstance(val, str):
                    outputs.append({"index": idx, "input_type": "string", "result": {"error": "value must be str"}})
                    continue
                result = self._predict_string(val)

            elif ftype == "image":
                if not isinstance(val, str):
                    outputs.append({"index": idx, "input_type": "image", "result": {"error": "value must be base64 str"}})
                    continue
                result = self._predict_image(val)

            else:
                outputs.append({
                    "index": idx,
                    "input_type": "unsupported",
                    "result": {"error": f"Unsupported feature type: {ftype}"},
                })
                continue

            outputs.append({
                "index": idx,
                "input_type": ftype,
                "result": result,
            })

        return outputs

# ---------- Example usage ----------

if __name__ == "__main__":
    # def encode_dummy_image():
    #     """Create a test image and encode as base64"""
    #     img = Image.new('RGB', (64, 64), color='red')
    #     buffered = BytesIO()
    #     img.save(buffered, format="JPEG")
    #     return base64.b64encode(buffered.getvalue()).decode()

    def load_bus_image():
        """Load bus.jpg and encode as base64"""
        with open("bus.jpg", "rb") as img_file:
            img_data = img_file.read()
            return base64.b64encode(img_data).decode()

    mm = MultiModalModel(load_vision=True, load_text=True)

    image_b64 = load_bus_image()

    features: List[Feature] = [
        {"type": "image",  "value": image_b64},
        {"type": "int",    "value": 7},
        {"type": "float",  "value": 2.5},
        {"type": "string", "value": "The capital of France is [MASK]."},
    ]
    
    print("Details:", mm.get_details())

    from pathlib import Path
    
    preds = mm.predict(features)

    # Keep a reference image on disk for plotting script
    img_filename = "bus.jpg"  # used by load_bus_image()
    img_pil = Image.open(img_filename).convert("RGB")
    W, H = img_pil.size

    # Collect & persist detections for the image feature
    det_threshold = 0.5
    detections: list[Detection] = []

    for p in preds:
        t = p["input_type"]
        r = p["result"]

        if t == "image" and isinstance(r, dict) and all(k in r for k in ("boxes", "labels", "scores")):
            filtered = [
                Detection(tuple(map(float, box)), int(label), float(score))
                for box, label, score in zip(r["boxes"], r["labels"], r["scores"])
                if float(score) > det_threshold
            ]
            detections.extend(filtered)
            print(f"[{p['index']}] image -> {len(filtered)} detections > {det_threshold}")
            # print(f"[{p['index']}] image -> {len(filtered)} detections")
            for d in filtered[:5]:
                print(f"  label_id={d.label}, score={d.score:.2f}, box={list(d.box)}")
        else:
            print(f"[{p['index']}] {t} -> {r}")

    # Save detections JSON (e.g., detections/bus.json)
    json_out = Path("detections") / f"{Path(img_filename).stem}.json"
    save_detections_json(
        image_path=img_filename,
        detections=detections,
        image_size=(W, H),
        save_path=str(json_out),
        extra={"source": "multi-feature-updated.py", "threshold": det_threshold}
        # extra={"source": "multi-feature-updated.py", "threshold": "None"}
    )
    print(f"Saved detections to: {json_out}")

    # print("Details:", mm.get_details())
    # preds = mm.predict(features)

    # for p in preds:
    #     t = p["input_type"]
    #     r = p["result"]
    #     if t == "image" and isinstance(r, dict) and all(k in r for k in ("boxes", "labels", "scores")):
    #         # Filter detections by confidence threshold
    #         filtered = [
    #             (box, label, score)
    #             for box, label, score in zip(r["boxes"], r["labels"], r["scores"])
    #             # if score > 0.5
    #         ]
    #         print(f"[{p['index']}] image -> {len(filtered)} detections > 0.5")
    #         for box, label, score in filtered[:5]:
    #             print(f"  label_id={label}, score={score:.2f}, box={box}")
    #     else:
    #         print(f"[{p['index']}] {t} -> {r}")

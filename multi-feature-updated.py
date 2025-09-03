import json
import os
from dataclasses import asdict, dataclass
from typing import Tuple, List, Any, Dict, Optional, TypedDict, Literal
from pathlib import Path

from transformers import pipeline
import torch
import base64
import numpy as np
from PIL import Image
from io import BytesIO

# ---------- Data structures ----------

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

def _encode_image_file_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _discover_images_in_dir(directory: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if not directory.exists():
        return []
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts])

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
    # Directory containing input images (read) and where per-image JSONs will be written.
    images_dir = Path("detections")

    # 1) Discover and encode all images under detections/
    image_paths: List[Path] = _discover_images_in_dir(images_dir)
    if not image_paths:
        print(f"[INFO] No images found in: {images_dir.resolve()}")
    else:
        print(f"[INFO] Found {len(image_paths)} images in: {images_dir.resolve()}")

    # Build features: all discovered images + the example numeric/text features
    features: List[Feature] = []

    # Maintain a mapping from feature index -> image path for post-processing
    feature_index_to_image_path: Dict[int, Path] = {}

    for p in image_paths:
        try:
            b64 = _encode_image_file_to_b64(p)
            feature_index_to_image_path[len(features)] = p
            features.append({"type": "image", "value": b64})
        except Exception as e:
            print(f"[WARN] Skipping {p.name}: {e}")

    # Add optional non-image features (kept from original example)
    features.extend([
        {"type": "int",    "value": 7},
        {"type": "float",  "value": 2.5},
        {"type": "string", "value": "The capital of France is [MASK]."},
    ])

    # 2) Initialize model and run predictions
    mm = MultiModalModel(load_vision=True, load_text=True)
    print("Details:", mm.get_details())

    preds = mm.predict(features)

    # 3) Collect & persist detections for each image feature
    det_threshold = 0.5

    for p in preds:
        idx = p["index"]
        t = p["input_type"]
        r = p["result"]

        if t != "image":
            # Log non-image results
            print(f"[{idx}] {t} -> {r}")
            continue

        # Only process if this index corresponds to an image we added
        img_path = feature_index_to_image_path.get(idx)
        if img_path is None:
            # Safety: image-like outputs without a mapped path
            print(f"[WARN] Image result at index {idx} has no mapped file path; skipping save.")
            continue

        if not (isinstance(r, dict) and all(k in r for k in ("boxes", "labels", "scores"))):
            print(f"[{idx}] image -> {r}")
            continue

        # Filter detections by threshold
        detections: list[Detection] = [
            Detection(tuple(map(float, box)), int(label), float(score))
            for box, label, score in zip(r["boxes"], r["labels"], r["scores"])
            if float(score) > det_threshold
        ]

        # Image size for metadata
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                W, H = im.size
        except Exception as e:
            print(f"[WARN] Could not read size for {img_path.name}: {e}")
            W = H = 0

        print(f"[{idx}] image {img_path.name} -> {len(detections)} detections > {det_threshold}")
        for d in detections[:5]:
            print(f"  label_id={d.label}, score={d.score:.2f}, box={list(d.box)}")

        # Save JSON alongside inputs in detections/<stem>.json
        json_out = images_dir / f"{img_path.stem}.json"
        save_detections_json(
            image_path=str(img_path),
            detections=detections,
            image_size=(W, H),
            save_path=str(json_out),
            extra={"source": "multi-feature-updated.py", "threshold": det_threshold}
        )
        print(f"Saved detections to: {json_out}")

"""
Multi-model inference router with human-readable COCO labels for FRCNN:
- {"integer": <int>}        -> classification (sklearn LogisticRegression)
- {"float": <float>}        -> regression (sklearn LinearRegression)
- {"image_base64": <str>}   -> object detection (torchvision Faster R-CNN, COCO labels)

Run:
    python multi_model_router_frcnn.py
"""

import base64
import io
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from PIL import Image
import base64

# -----------------------------
# Classic ML: sklearn
# -----------------------------
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

# -----------------------------
# DL: PyTorch / Torchvision
# -----------------------------
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# ------------------------------------------------------------
# COCO class names (index-aligned with torchvision models)
# 0 is background (not returned by FRCNN); 1..90 are classes.
# ------------------------------------------------------------
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]
# Convenience map: id -> name
COCO_ID2NAME = {i: name for i, name in enumerate(COCO_CLASSES)}


# ============================================================
#                 REGRESSION / CLASSIFICATION
# ============================================================

def train_regression_model(seed: int = 42):
    """Simple 1D regression: y = 3x + sin(x) + noise"""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-10, 10, size=(2000, 1))
    y = 3 * X[:, 0] + np.sin(X[:, 0]) + rng.normal(0, 0.5, size=2000)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    reg = LinearRegression()
    reg.fit(Xtr, ytr)

    ypred = reg.predict(Xte)
    metrics = {
        "mse": float(mean_squared_error(yte, ypred)),
        "r2": float(r2_score(yte, ypred)),
    }
    return reg, metrics


def train_classification_model(seed: int = 42):
    """Simple 1D binary classification: class = 1 if x >= 0 else 0"""
    rng = np.random.default_rng(seed)
    X = rng.integers(-50, 50, size=(3000, 1))
    y = (X[:, 0] >= 0).astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    clf = LogisticRegression()
    clf.fit(Xtr, ytr)

    ypred = clf.predict(Xte)
    metrics = {
        "accuracy": float(accuracy_score(yte, ypred)),
        "f1": float(f1_score(yte, ypred)),
    }
    return clf, metrics


# ============================================================
#                        FRCNN ASSETS
# ============================================================

@dataclass
class FRCNNAssets:
    model: torch.nn.Module
    device: torch.device
    transform: transforms.Compose


def load_frcnn_assets() -> FRCNNAssets:
    """
    Load Faster R-CNN ResNet50 FPN pretrained on COCO.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),  # converts PIL [H,W,C] uint8 -> FloatTensor [C,H,W] in [0,1]
    ])
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device)
    model.eval()
    return FRCNNAssets(model=model, device=device, transform=transform)


def b64_to_tensor_image(img_b64: str, transform: transforms.Compose, device: torch.device) -> torch.Tensor:
    """
    Decode base64 -> PIL.Image -> torch.Tensor [C,H,W] in [0,1]
    """
    img_bytes = base64.b64decode(img_b64)
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = transform(pil).to(device)  # [3,H,W]
    return x


# ============================================================
#                     INFERENCE ROUTER
# ============================================================

@dataclass
class Router:
    reg_model: Any
    clf_model: Any
    frcnn: FRCNNAssets
    det_thresh: float = 0.5  # detection confidence threshold

    def predict(self, inferences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for item in inferences:
            if "integer" in item:
                x = np.array([[float(item["integer"])]], dtype=np.float32)
                yhat = int(self.clf_model.predict(x)[0])
                # optional: proba if available
                proba = float(self.clf_model.predict_proba(x)[0][yhat])
                results.append({
                    "type": "classification",
                    "input": item["integer"],
                    "pred_class": yhat,
                    "confidence": round(proba, 4)
                })

            elif "float" in item:
                x = np.array([[float(item["float"])]], dtype=np.float32)
                yhat = float(self.reg_model.predict(x)[0])
                results.append({
                    "type": "regression",
                    "input": item["float"],
                    "pred_value": round(yhat, 4)
                })

            elif "image_base64" in item:
                img_t = b64_to_tensor_image(item["image_base64"], self.frcnn.transform, self.frcnn.device)
                with torch.no_grad():
                    pred = self.frcnn.model([img_t])[0]  # FRCNN expects list[Tensor]
                dets = []
                for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
                    s = float(score.item())
                    if s < self.det_thresh:
                        continue
                    lid = int(label.item())
                    name = COCO_ID2NAME.get(lid, f"cls_{lid}")
                    dets.append({
                        "label": name,                    # human-readable COCO label
                        "label_id": lid,                  # numeric id
                        "bbox": [float(v) for v in box.tolist()],  # [x1,y1,x2,y2]
                        "score": round(s, 4)
                    })
                results.append({
                    "type": "object_detection",
                    "detections": dets
                })

            else:
                results.append({"error": f"Unsupported key(s): {list(item.keys())}"})
        return results


# ============================================================
#                        MAIN (demo)
# ============================================================

def main():
    # Convert image to base64 string
    def image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode("utf-8")  # convert bytes → string

    # Example usage
    img_path = "./detections/1.jpg"
    base64_string = image_to_base64(img_path)

    # 1) Train light ML models
    reg_model, reg_metrics = train_regression_model()
    print("[REG] metrics:", reg_metrics)

    clf_model, clf_metrics = train_classification_model()
    print("[CLF] metrics:", clf_metrics)

    # 2) Load DL detector
    frcnn_assets = load_frcnn_assets()
    print("[DL ] model: Faster R-CNN ResNet50 FPN (COCO)")

    # 3) Build router
    router = Router(reg_model=reg_model, clf_model=clf_model, frcnn=frcnn_assets, det_thresh=0.5)

    # 4) Example inferences
    # NOTE: put a real base64-encoded image (RGB JPG/PNG) in example_b64 to test detection.
    example_b64 = base64_string  # replace with your own base64 image string

    inferences = [
        {"integer": 7},             # -> classification
        {"integer": -4},            # -> classification
        {"float": 3.1415},          # -> regression
        {"float": -2.0},            # -> regression
        {"image_base64": example_b64},  # -> object detection
    ]

    results = router.predict(inferences)
    for r in results:
        print("ROUTED:", r)

    if example_b64 is not None:
        det_res = router.predict([{"image_base64": example_b64}])[0]
        print("ROUTED (DL):", det_res)


if __name__ == "__main__":
    main()

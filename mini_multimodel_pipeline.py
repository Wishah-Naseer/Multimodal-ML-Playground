"""
Multi-model inference router with human-readable COCO labels for FRCNN,
NOW WITH optional Faster R-CNN fine-tuning (COCO-format dataset).

This script implements a unified inference pipeline that can handle three types of inputs:
1. Integer inputs -> Binary classification using sklearn LogisticRegression
2. Float inputs -> Regression using sklearn LinearRegression  
3. Base64 encoded images -> Object detection using PyTorch Faster R-CNN

Input routing:
- {"integer": <int>}        -> classification (sklearn LogisticRegression)
- {"float": <float>}        -> regression (sklearn LinearRegression)
- {"image_base64": <str>}   -> object detection (torchvision Faster R-CNN)

Run (inference-only as before):
    python multi_model_router_frcnn.py

Run (fine-tune then infer; requires COCO-format dataset + pycocotools):
    python multi_model_router_frcnn.py --train-images ./data/train/images \
                                       --train-ann ./data/train/annotations.json \
                                       --val-images ./data/val/images \
                                       --val-ann ./data/val/annotations.json \
                                       --epochs 5 --batch-size 2
"""

import argparse
import base64
import io
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

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
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Try COCO tooling (optional)
try:
    from pycocotools.coco import COCO  # type: ignore
    _HAS_COCO = True
except Exception:
    _HAS_COCO = False

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
# Create a mapping from class ID to class name for easy lookup
COCO_ID2NAME = {i: name for i, name in enumerate(COCO_CLASSES)}

# ============================================================
#                 REGRESSION / CLASSIFICATION
# ============================================================

def train_regression_model(seed: int = 42):
    """
    Train a simple linear regression model on synthetic data.
    
    Creates synthetic data following the pattern: y = 3x + sin(x) + noise
    This demonstrates a non-linear relationship that linear regression can approximate.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (trained_model, evaluation_metrics)
    """
    # Set random seed for reproducible results
    rng = np.random.default_rng(seed)
    
    # Generate synthetic data: 2000 samples from uniform distribution [-10, 10]
    X = rng.uniform(-10, 10, size=(2000, 1))
    
    # Create target values: y = 3x + sin(x) + Gaussian noise
    y = 3 * X[:, 0] + np.sin(X[:, 0]) + rng.normal(0, 0.5, size=2000)

    # Split data into training (80%) and test (20%) sets
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # Train linear regression model
    reg = LinearRegression()
    reg.fit(Xtr, ytr)

    # Evaluate model performance
    ypred = reg.predict(Xte)
    metrics = {
        "mse": float(mean_squared_error(yte, ypred)),  # Mean squared error
        "r2": float(r2_score(yte, ypred)),            # R-squared coefficient
    }
    return reg, metrics


def train_classification_model(seed: int = 42):
    """
    Train a simple binary classification model on synthetic data.
    
    Creates synthetic data where class = 1 if x >= 0 else 0
    This creates a simple decision boundary at x = 0.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (trained_model, evaluation_metrics)
    """
    # Set random seed for reproducible results
    rng = np.random.default_rng(seed)
    
    # Generate synthetic data: 3000 integer samples from [-50, 50]
    X = rng.integers(-50, 50, size=(3000, 1))
    
    # Create binary labels: 1 if x >= 0, 0 otherwise
    y = (X[:, 0] >= 0).astype(int)

    # Split data into training (80%) and test (20%) sets
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # Train logistic regression model
    clf = LogisticRegression()
    clf.fit(Xtr, ytr)

    # Evaluate model performance
    ypred = clf.predict(Xte)
    metrics = {
        "accuracy": float(accuracy_score(yte, ypred)),  # Classification accuracy
        "f1": float(f1_score(yte, ypred)),             # F1 score (harmonic mean of precision/recall)
    }
    return clf, metrics

# ============================================================
#                        FRCNN DATASET
# ============================================================

class CocoDetectionDataset(Dataset):
    """
    COCO-format detection dataset loader for PyTorch.
    
    This class handles loading images and annotations from a COCO-format dataset
    and converts them to the format expected by PyTorch's detection models.
    
    Attributes:
        images_dir: Directory containing the image files
        coco: COCO annotation object
        ids: List of image IDs
        train: Whether this is training data (affects augmentation)
        tf: Basic transform (PIL to tensor)
        aug: Augmentation transform (horizontal flip)
        cat_id_to_name: Mapping from category ID to name
        cat_ids: List of category IDs in the dataset
        cat_id_to_contig: Mapping from original category ID to contiguous ID
        contig_to_name: Mapping from contiguous ID to category name
        num_classes: Total number of classes (including background)
    """
    def __init__(self, images_dir: str, ann_file: str, train: bool = True):
        # Check if pycocotools is available
        if not _HAS_COCO:
            raise RuntimeError("pycocotools is required to use CocoDetectionDataset.")
        
        self.images_dir = images_dir
        self.coco = COCO(ann_file)  # Load COCO annotations
        self.ids = list(self.coco.imgs.keys())  # Get all image IDs
        self.train = train
        
        # Basic transform: convert PIL image to tensor
        self.tf = transforms.Compose([transforms.ToTensor()])
        
        # Minimal augmentation: only horizontal flip with 50% probability
        # This can be extended with more augmentations as needed
        self.aug = transforms.RandomHorizontalFlip(p=0.5)

        # Build class id → name mapping (COCO-style, contiguous IDs not guaranteed)
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_id_to_name = {c["id"]: c["name"] for c in cats}

        # Map dataset category ids to a contiguous set starting at 1 (as TorchVision expects)
        # Background class is 0, so custom classes start at 1
        self.cat_ids = sorted(self.cat_id_to_name.keys())
        self.cat_id_to_contig = {cid: i + 1 for i, cid in enumerate(self.cat_ids)}
        self.contig_to_name = {i + 1: self.cat_id_to_name[cid] for i, cid in enumerate(self.cat_ids)}
        self.num_classes = len(self.cat_ids) + 1  # +1 for background class

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.ids)

    def _load_image(self, img_id: int) -> Image.Image:
        """
        Load and return a PIL Image for the given image ID.
        
        Args:
            img_id: COCO image ID
            
        Returns:
            PIL Image in RGB format
        """
        info = self.coco.loadImgs(img_id)[0]  # Get image info
        path = os.path.join(self.images_dir, info["file_name"])  # Construct file path
        img = Image.open(path).convert("RGB")  # Load and convert to RGB
        return img

    def _load_target(self, img_id: int) -> Dict[str, Any]:
        """
        Load annotations for the given image ID and convert to PyTorch format.
        
        Args:
            img_id: COCO image ID
            
        Returns:
            Dictionary containing target information in PyTorch format
        """
        # Get annotation IDs for this image
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)  # Load annotations

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for a in anns:
            # COCO bbox format is [x, y, w, h] → convert to [x1, y1, x2, y2]
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])  # Convert to corner format
            
            # Map original category ID to contiguous ID
            labels.append(self.cat_id_to_contig[a["category_id"]])
            
            # Get area (use bbox area if not provided)
            areas.append(a.get("area", w * h))
            
            # Get crowd flag (default to 0 if not provided)
            iscrowd.append(a.get("iscrowd", 0))

        # Create target dictionary in PyTorch format
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),      # Bounding boxes
            "labels": torch.as_tensor(labels, dtype=torch.int64),      # Class labels
            "image_id": torch.tensor([img_id]),                       # Image ID
            "area": torch.as_tensor(areas, dtype=torch.float32),      # Box areas
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),   # Crowd flags
        }
        return target

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (image_tensor, target_dict)
        """
        img_id = self.ids[idx]
        img = self._load_image(img_id)
        target = self._load_target(img_id)

        # Convert PIL image to tensor
        img = self.tf(img)

        # Apply augmentation only during training
        if self.train:
            if isinstance(self.aug, transforms.RandomHorizontalFlip):
                # Apply horizontal flip with probability p
                if torch.rand(1).item() < self.aug.p:
                    _, h, w = img.shape
                    # Flip the image along the width dimension
                    img = torch.flip(img, dims=[2])
                    
                    # Update bounding boxes to match the flipped image
                    boxes = target["boxes"]
                    x1 = boxes[:, 0].clone()
                    x2 = boxes[:, 2].clone()
                    boxes[:, 0] = w - x2  # New x1 = width - old x2
                    boxes[:, 2] = w - x1  # New x2 = width - old x1
                    target["boxes"] = boxes

        return img, target


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    
    PyTorch detection models expect batches as (List[images], List[targets])
    rather than stacked tensors.
    
    Args:
        batch: List of (image, target) tuples
        
    Returns:
        Tuple of (List[images], List[targets])
    """
    return tuple(zip(*batch))  # List[(img, target)] -> (List[img], List[target])

# ============================================================
#                        FRCNN ASSETS
# ============================================================

@dataclass
class FRCNNAssets:
    """
    Container class for Faster R-CNN model and related assets.
    
    Attributes:
        model: The Faster R-CNN model
        device: Device to run the model on (CPU/GPU)
        transform: Image preprocessing transform
        id2name: Mapping from class ID to class name
        score_thresh: Confidence threshold for displaying detections
    """
    model: torch.nn.Module
    device: torch.device
    transform: transforms.Compose
    id2name: Dict[int, str]
    score_thresh: float = 0.5  # display threshold (post-NMS)

def get_frcnn_model(num_classes: Optional[int],
                    finetune: bool,
                    weights: Optional[str] = "DEFAULT",
                    weights_backbone: Optional[str] = "IMAGENET1K_V1") -> nn.Module:
    """
    Build and configure Faster R-CNN model.
    
    Args:
        num_classes: Number of classes (including background). If None, use COCO defaults.
        finetune: Whether to prepare model for fine-tuning
        weights: Model weights to load
        weights_backbone: Backbone weights to load
        
    Returns:
        Configured Faster R-CNN model
    """
    # Load pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=weights_backbone)
    
    if finetune and (num_classes is not None):
        # Replace classification head for custom number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def load_frcnn_assets(num_classes: Optional[int] = None,
                      finetune: bool = False,
                      score_thresh: float = 0.5,
                      id2name: Optional[Dict[int, str]] = None) -> FRCNNAssets:
    """
    Load and configure Faster R-CNN model and related assets.
    
    Args:
        num_classes: Number of classes for fine-tuning
        finetune: Whether to prepare for fine-tuning
        score_thresh: Confidence threshold for detections
        id2name: Class ID to name mapping
        
    Returns:
        FRCNNAssets object containing model and configuration
    """
    # Determine device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Basic transform: convert PIL to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Build model
    model = get_frcnn_model(num_classes=num_classes, finetune=finetune)
    model.to(device)
    
    # Set model mode (train for fine-tuning, eval for inference)
    model.train() if finetune else model.eval()

    # Use provided mapping if training on custom dataset; else fallback to COCO
    id2name = id2name if id2name is not None else COCO_ID2NAME
    
    return FRCNNAssets(model=model, device=device, transform=transform, id2name=id2name, score_thresh=score_thresh)

def b64_to_tensor_image(img_b64: str, transform: transforms.Compose, device: torch.device) -> torch.Tensor:
    """
    Convert base64 encoded image string to PyTorch tensor.
    
    Args:
        img_b64: Base64 encoded image string
        transform: Image preprocessing transform
        device: Device to place tensor on
        
    Returns:
        Image tensor in format [C, H, W] with values in [0, 1]
    """
    # Decode base64 string to bytes
    img_bytes = base64.b64decode(img_b64)
    
    # Convert bytes to PIL Image
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # Apply transform and move to device
    x = transform(pil).to(device)  # [3, H, W]
    return x

# ============================================================
#                   FRCNN TRAIN/EVAL PIPELINE
# ============================================================

def train_frcnn(model: nn.Module,
                train_loader: DataLoader,
                val_loader: Optional[DataLoader],
                device: torch.device,
                epochs: int = 5,
                lr: float = 0.005,
                weight_decay: float = 0.0005,
                lr_step_size: int = 3,
                lr_gamma: float = 0.1) -> None:
    """
    Train/fine-tune Faster R-CNN model.
    
    This function implements a minimal training loop for object detection.
    It handles both classification/regression losses and RPN (Region Proposal Network) losses.
    
    Args:
        model: Faster R-CNN model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        device: Device to train on
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        lr_step_size: Learning rate scheduler step size
        lr_gamma: Learning rate scheduler gamma
    """
    # Get trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Setup optimizer (SGD with momentum)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    
    # Setup learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    # Set model to training mode
    model.train()
    
    # Training loop
    for ep in range(epochs):
        ep_loss = 0.0
        
        # Iterate over training batches
        for images, targets in train_loader:
            # Move data to device
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass: returns dictionary of losses
            loss_dict = model(images, targets)
            
            # Sum all losses
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # Accumulate epoch loss
            ep_loss += float(losses.item())

        # Step learning rate scheduler
        lr_scheduler.step()
        
        # Calculate average loss for the epoch
        avg_loss = ep_loss / max(1, len(train_loader))
        print(f"[TRAIN] epoch {ep+1}/{epochs} avg_loss={avg_loss:.4f}")

        # Optional validation pass
        if val_loader is not None:
            model.eval()  # Set to evaluation mode
            with torch.no_grad():  # Disable gradient computation
                val_loss = 0.0
                for images, targets in val_loader:
                    # Move data to device
                    images = list(img.to(device) for img in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    
                    # Forward pass
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    val_loss += float(losses.item())
                
                # Calculate average validation loss
                val_avg = val_loss / max(1, len(val_loader))
            print(f"[VAL  ] epoch {ep+1}/{epochs} avg_loss={val_avg:.4f}")
            model.train()  # Set back to training mode

    print("[TRAIN] done.")

# ============================================================
#                     INFERENCE ROUTER
# ============================================================

@dataclass
class Router:
    """
    Multi-model inference router that handles different input types.
    
    This class routes inputs to appropriate models based on the input type:
    - Integer inputs -> Classification model
    - Float inputs -> Regression model  
    - Base64 image inputs -> Object detection model
    
    Attributes:
        reg_model: Trained regression model
        clf_model: Trained classification model
        frcnn: Faster R-CNN model and assets
    """
    reg_model: Any
    clf_model: Any
    frcnn: FRCNNAssets

    def predict(self, inferences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Route inputs to appropriate models and return predictions.
        
        Args:
            inferences: List of input dictionaries, each containing one of:
                       - {"integer": <int>} for classification
                       - {"float": <float>} for regression
                       - {"image_base64": <str>} for object detection
                       
        Returns:
            List of prediction results
        """
        results = []
        
        for item in inferences:
            # Handle integer input -> classification
            if "integer" in item:
                # Prepare input for sklearn model (2D array)
                x = np.array([[float(item["integer"])]], dtype=np.float32)
                
                # Get prediction and probability
                yhat = int(self.clf_model.predict(x)[0])
                proba = float(self.clf_model.predict_proba(x)[0][yhat])
                
                results.append({
                    "type": "classification",
                    "input": item["integer"],
                    "pred_class": yhat,
                    "confidence": round(proba, 4)
                })

            # Handle float input -> regression
            elif "float" in item:
                # Prepare input for sklearn model (2D array)
                x = np.array([[float(item["float"])]], dtype=np.float32)
                
                # Get prediction
                yhat = float(self.reg_model.predict(x)[0])
                
                results.append({
                    "type": "regression",
                    "input": item["float"],
                    "pred_value": round(yhat, 4)
                })

            # Handle base64 image input -> object detection
            elif "image_base64" in item:
                # Convert base64 to tensor
                img_t = b64_to_tensor_image(item["image_base64"], self.frcnn.transform, self.frcnn.device)
                
                # Run inference
                with torch.no_grad():
                    self.frcnn.model.eval()  # ensure eval mode for inference
                    pred = self.frcnn.model([img_t])[0]  # FRCNN expects list[Tensor]
                
                dets = []
                # Process detections
                # NOTE on thresholding:
                # TorchVision runs NMS internally; here we only apply a *display* score threshold.
                for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
                    s = float(score.item())
                    
                    # Skip detections below threshold
                    if s < self.frcnn.score_thresh:
                        continue
                    
                    lid = int(label.item())
                    name = self.frcnn.id2name.get(lid, f"cls_{lid}")
                    
                    dets.append({
                        "label": name,                                  # human-readable label
                        "label_id": lid,                                # numeric id
                        "bbox": [float(v) for v in box.tolist()],       # [x1,y1,x2,y2]
                        "score": round(s, 4)
                    })
                
                results.append({
                    "type": "object_detection",
                    "detections": dets
                })

            # Handle unsupported input types
            else:
                results.append({"error": f"Unsupported key(s): {list(item.keys())}"})
        
        return results

# ============================================================
#                        MAIN (demo)
# ============================================================

def image_to_base64(image_path: str) -> str:
    """
    Convert image file to base64 encoded string.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded string
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")

def maybe_build_coco_loaders(train_images: Optional[str],
                             train_ann: Optional[str],
                             val_images: Optional[str],
                             val_ann: Optional[str],
                             batch_size: int,
                             num_workers: int = 2) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[int], Optional[Dict[int, str]]]:
    """
    Build COCO dataset loaders if dataset paths are provided.
    
    This function conditionally creates training and validation data loaders
    for fine-tuning the object detection model.
    
    Args:
        train_images: Path to training images directory
        train_ann: Path to training annotations file
        val_images: Path to validation images directory  
        val_ann: Path to validation annotations file
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, num_classes, id2name) or 
        (None, None, None, None) if dataset not provided
    """
    # Check if COCO paths provided but pycocotools not available
    if (train_images and train_ann) and not _HAS_COCO:
        raise RuntimeError("COCO paths provided but pycocotools not installed.")

    # Build loaders if training data is provided
    if train_images and train_ann and _HAS_COCO:
        # Create training dataset
        train_ds = CocoDetectionDataset(train_images, train_ann, train=True)
        
        # Create validation dataset if validation data provided
        val_ds = CocoDetectionDataset(val_images, val_ann, train=False) if (val_images and val_ann) else None

        # Create data loaders
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                              num_workers=num_workers, collate_fn=collate_fn) if val_ds else None

        # Get dataset information
        num_classes = train_ds.num_classes
        id2name = train_ds.contig_to_name  # contiguous id→name (1..K), 0 is background
        return train_loader, val_loader, num_classes, id2name

    return None, None, None, None

def main():
    """
    Main function that orchestrates the entire pipeline.
    
    This function:
    1. Parses command line arguments
    2. Trains lightweight ML models (regression and classification)
    3. Optionally sets up COCO dataset for fine-tuning
    4. Loads and optionally fine-tunes Faster R-CNN
    5. Creates the inference router
    6. Runs example inferences
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-images", type=str, default=None, 
                       help="Path to training images directory (COCO format)")
    parser.add_argument("--train-ann", type=str, default=None, 
                       help="Path to training annotations.json (COCO format)")
    parser.add_argument("--val-images", type=str, default=None, 
                       help="Path to val images directory (COCO format)")
    parser.add_argument("--val-ann", type=str, default=None, 
                       help="Path to val annotations.json (COCO format)")
    parser.add_argument("--epochs", type=int, default=0, 
                       help="Number of fine-tuning epochs; 0 = skip training")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--score-thresh", type=float, default=0.5, 
                       help="Display score threshold (post-NMS)")
    parser.add_argument("--example-image", type=str, default="./detections/1.jpg", 
                       help="Example image path for demo inference")
    args = parser.parse_args()

    # Step 1: Train lightweight ML models
    print("Training lightweight ML models...")
    reg_model, reg_metrics = train_regression_model()
    print("[REG] metrics:", reg_metrics)

    clf_model, clf_metrics = train_classification_model()
    print("[CLF] metrics:", clf_metrics)

    # Step 2: Build (optional) COCO dataloaders for fine-tuning
    print("Setting up dataset loaders...")
    train_loader, val_loader, num_classes, id2name = maybe_build_coco_loaders(
        args.train_images, args.train_ann, args.val_images, args.val_ann, 
        batch_size=args.batch_size
    )

    # Step 3: Build DL detector (pretrained; optionally adapt head for finetune)
    print("Loading Faster R-CNN model...")
    finetune = (args.epochs > 0) and (train_loader is not None) and (num_classes is not None)
    frcnn_assets = load_frcnn_assets(
        num_classes=num_classes if finetune else None,
        finetune=finetune,
        score_thresh=args.score_thresh,
        id2name=id2name if finetune else None
    )

    # Step 4: Fine-tune model if requested
    if finetune:
        print(f"[DL ] model: Faster R-CNN ResNet50 FPN (fine-tuning, num_classes={num_classes})")
        train_frcnn(frcnn_assets.model, train_loader, val_loader, frcnn_assets.device, epochs=args.epochs)
        frcnn_assets.model.eval()  # Set to evaluation mode after training
    else:
        print("[DL ] model: Faster R-CNN ResNet50 FPN (COCO pretrained, inference-only)")

    # Step 5: Build the inference router
    print("Creating inference router...")
    router = Router(reg_model=reg_model, clf_model=clf_model, frcnn=frcnn_assets)

    # Step 6: Run example inferences
    print("Running example inferences...")
    
    # Prepare example image for detection demo
    example_b64 = image_to_base64(args.example_image) if os.path.isfile(args.example_image) else None

    # Create test inputs
    inferences = [
        {"integer": 7},             # -> classification (should predict class 1)
        {"integer": -4},           # -> classification (should predict class 0)
        {"float": 3.1415},         # -> regression
        {"float": -2.0},           # -> regression
    ]
    
    # Add image detection if example image exists
    if example_b64 is not None:
        inferences.append({"image_base64": example_b64})

    # Run inference through router
    results = router.predict(inferences)
    for r in results:
        print("ROUTED:", r)

    # Additional detailed output for object detection
    if example_b64 is not None:
        det_res = router.predict([{"image_base64": example_b64}])[0]
        print("ROUTED (DL):", det_res)


if __name__ == "__main__":
    main()
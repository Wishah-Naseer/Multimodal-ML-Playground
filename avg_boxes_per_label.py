# python avg_boxes_per_label.py --json detections\bus.json --out detections\bus_avg.json --score-thres 0.5
import argparse, json, os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -------- JSON I/O --------

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

# -------- Robust detection extraction (works with your files and raw Ollama shapes) --------

def _first_numeric_key(d: Dict[str, Any]) -> Optional[str]:
    ks = [k for k in d.keys() if isinstance(k, str) and k.isdigit()]
    return str(min(map(int, ks))) if ks else None

def _extract_vision_block(obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(obj, dict):
        if "result" in obj and isinstance(obj["result"], dict) and "boxes" in obj["result"]:
            return obj["result"]
        k = _first_numeric_key(obj)
        if k and isinstance(obj[k], dict) and "boxes" in obj[k]:
            return obj[k]
        if "boxes" in obj:
            return obj
    if isinstance(obj, list):
        for it in obj:
            vb = _extract_vision_block(it)
            if vb is not None:
                return vb
    return None

def adapt_to_detections(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    dets = payload.get("detections")
    if isinstance(dets, list) and dets:
        return dets
    vision = _extract_vision_block(payload)
    if not vision:
        return []
    boxes  = vision.get("boxes", []) or []
    labels = vision.get("labels", []) or []
    scores = vision.get("scores", []) or []
    out: List[Dict[str, Any]] = []
    for i, b in enumerate(boxes):
        if not isinstance(b, (list, tuple)) or len(b) < 4:
            continue
        out.append({
            "box": [float(v) for v in b[:4]],
            "label": labels[i] if i < len(labels) else None,
            "score": (float(scores[i]) if i < len(scores) and scores[i] is not None else None),
        })
    return out

# -------- Geometry / clustering --------

def iou(b1: List[float], b2: List[float]) -> float:
    x11,y11,x12,y12 = b1
    x21,y21,x22,y22 = b2
    xi1, yi1 = max(x11,x21), max(y11,y21)
    xi2, yi2 = min(x12,x22), min(y12,y22)
    iw, ih = max(0.0, xi2-xi1), max(0.0, yi2-yi1)
    inter = iw * ih
    if inter <= 0: return 0.0
    a1 = max(0.0, x12-x11) * max(0.0, y12-y11)
    a2 = max(0.0, x22-x21) * max(0.0, y22-y21)
    return inter / max(a1 + a2 - inter, 1e-9)

def weighted_avg_box(items: List[Dict[str, Any]]) -> Tuple[List[float], float]:
    """Return (box, score) where box coords are score-weighted averages; score=max(scores)."""
    if not items:
        return [0,0,0,0], 0.0
    wsum = sum(d["score"] for d in items)
    if wsum <= 0:
        # fallback to uniform average
        x1 = sum(d["box"][0] for d in items) / len(items)
        y1 = sum(d["box"][1] for d in items) / len(items)
        x2 = sum(d["box"][2] for d in items) / len(items)
        y2 = sum(d["box"][3] for d in items) / len(items)
    else:
        x1 = sum(d["box"][0]*d["score"] for d in items) / wsum
        y1 = sum(d["box"][1]*d["score"] for d in items) / wsum
        x2 = sum(d["box"][2]*d["score"] for d in items) / wsum
        y2 = sum(d["box"][3]*d["score"] for d in items) / wsum
    score = max(d["score"] for d in items)
    return [x1,y1,x2,y2], score

def cluster_by_iou(items: List[Dict[str, Any]], thr: float) -> List[List[Dict[str, Any]]]:
    """Greedy clustering by IoU threshold (within the same label)."""
    clusters: List[List[Dict[str, Any]]] = []
    for d in items:
        placed = False
        for c in clusters:
            # compare to current cluster representative (first member)
            if iou(d["box"], c[0]["box"]) >= thr:
                c.append(d)
                placed = True
                break
        if not placed:
            clusters.append([d])
    return clusters

# -------- Main averaging logic --------

def canon_label(lbl: Any) -> str:
    return str(lbl) if lbl is not None else ""

def preprocess_scores(dets: List[Dict[str, Any]], score_thres: float, score_default: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in dets:
        s = d.get("score")
        try:
            s = float(s) if s is not None else score_default
        except Exception:
            s = score_default
        if s >= score_thres:
            e = dict(d)
            e["score"] = s
            out.append(e)
    # sort high→low (not strictly required, but stable)
    out.sort(key=lambda x: x["score"], reverse=True)
    return out

def average_per_label(
    dets: List[Dict[str, Any]],
    cluster_iou: float = 0.0,
) -> List[Dict[str, Any]]:
    """Average boxes per label; if cluster_iou>0, cluster within label first, then average each cluster."""
    # group by label
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for d in dets:
        groups.setdefault(canon_label(d.get("label")), []).append(d)

    fused: List[Dict[str, Any]] = []
    for label, items in groups.items():
        if cluster_iou > 0:
            clusters = cluster_by_iou(items, cluster_iou)
            for c in clusters:
                box, s = weighted_avg_box(c)
                fused.append({"box": box, "label": label, "score": s, "count": len(c)})
        else:
            box, s = weighted_avg_box(items)
            fused.append({"box": box, "label": label, "score": s, "count": len(items)})
    return fused

# -------- CLI --------

def main():
    ap = argparse.ArgumentParser(description="Average bounding boxes per label (optionally cluster by IoU first).")
    ap.add_argument("--json", required=True, help="Input detections JSON (e.g., detections/bus.json).")
    ap.add_argument("--out", default=None, help="Output JSON path. Default: <stem>_avg.json next to input.")
    ap.add_argument("--score-thres", type=float, default=0.3, help="Drop boxes with score below this.")
    ap.add_argument("--score-default", type=float, default=0.9, help="Score to assign when score is null.")
    ap.add_argument("--cluster-iou", type=float, default=0.0,
                    help="If >0, cluster same-label boxes by IoU before averaging (e.g., 0.55).")
    args = ap.parse_args()

    payload = load_json(args.json)
    dets = adapt_to_detections(payload)
    dets = preprocess_scores(dets, score_thres=args.score_thres, score_default=args.score_default)
    fused = average_per_label(dets, cluster_iou=args.cluster_iou)

    out_path = args.out or str(Path(args.json).with_name(f"{Path(args.json).stem}_avg.json"))

    # write same schema back
    out_payload: Dict[str, Any] = {
        "image": payload.get("image"),
        "image_size": payload.get("image_size"),
        "detections": [{"box": d["box"], "label": d["label"], "score": d["score"]} for d in fused],
        "meta": {
            **payload.get("meta", {}),
            "preprocess": {
                "method": "average_per_label",
                "cluster_iou": args.cluster_iou,
                "score_thres": args.score_thres,
                "score_default": args.score_default,
            },
            "counts_per_label": {canon_label(k): len(v) for k, v in
                                 {canon_label(x.get('label')): [] for x in dets}.items()}
        }
    }
    save_json(out_path, out_payload)
    print(f"Saved averaged detections -> {out_path} ({len(fused)} boxes)")

if __name__ == "__main__":
    main()

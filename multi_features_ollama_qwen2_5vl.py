import json, os
from typing import Any, Tuple
from PIL import Image

def save_detections_json(
    image_path: str,
    detections: list[dict[str, Any]],
    image_size: Tuple[int, int],
    save_path: str,
    extra: dict | None = None
) -> None:
    payload = {
        "image": os.path.basename(image_path),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "detections": detections,  # [{"box":[x1,y1,x2,y2], "label": <str|int|None>, "score": <float|None>}, ...]
        "meta": extra or {},
    }
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


import json
import re
import requests
import base64
import os
from typing import List, TypedDict, Literal, Any

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5vl")  

FeatureType = Literal["image", "string", "int", "float"]

class Feature(TypedDict):
    type: FeatureType
    value: Any

def extract_first_balanced_json(s: str) -> str:
    """Return the first balanced top-level JSON object in s, or ''."""
    start = s.find("{")
    if start < 0:
        return ""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return ""

def build_prompt_and_images(features: List[Feature]):
    """
    Build a single instruction for the model and collect images as base64 strings.
    Qwen2.5-VL supports multiple images in one message via the 'images' field.
    """
    lines = [
        "You will receive typed features. Produce ONE JSON object mapping index->result.",
        "Rules per type:",
        ' - image: detect objects and return {"labels":[...],"boxes":[[x1,y1,x2,y2],...]}. If detection is unavailable, return {"caption":"<short>"}',
        ' - string: if it contains [MASK], respond {"answer":"<best single word>"}. Otherwise respond {"answer":"<short reply>"}',
        ' - int: respond {"y": 2*x}',
        ' - float: respond {"y": round(x^2, 1)}',
        "Output strict JSON only (no code fences, no prose)."
    ]

    images_b64: List[str] = []
    image_slot_map = {}  # feature index -> image index in images_b64

    for i, f in enumerate(features):
        t = f.get("type")
        v = f.get("value")
        if t == "image":
            if isinstance(v, str):
                image_slot_map[i] = len(images_b64)
                images_b64.append(v)  # base64 (no data URL prefix)
                lines.append(f"- index {i}: image <image_{image_slot_map[i]}>")
            else:
                lines.append(f"- index {i}: image INVALID")
        elif t == "string":
            s = str(v)
            if "[MASK]" in s:
                lines.append(f'- index {i}: string (fill [MASK]) -> "{s}"')
            else:
                lines.append(f'- index {i}: string -> "{s}"')
        elif t == "int":
            lines.append(f"- index {i}: int x={int(v)}")
        elif t == "float":
            lines.append(f"- index {i}: float x={float(v)}")
        else:
            lines.append(f"- index {i}: unsupported type '{t}'")

    user_prompt = "\n".join(lines)
    return user_prompt, images_b64

def call_ollama_qwen_vl(features: List[Feature]) -> dict:
    prompt, images_b64 = build_prompt_and_images(features)

    # Ollama /api/chat supports messages with images on each message object:
    # {"role":"user","content":"...", "images":["<base64>", ...]}
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise function that outputs strict JSON only. Begin with '{' and end with '}'."
            },
            {
                "role": "user",
                "content": prompt,
                # attach all images seen in the prompt, in the same order
                **({"images": images_b64} if images_b64 else {})
            }
        ]
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    # Ollama /api/chat returns {"message":{"content": "..."} , ...}
    content = data.get("message", {}).get("content", "")
    json_str = extract_first_balanced_json(content)
    if not json_str:
        # try to strip any code fences just in case
        cleaned = re.sub(r"```.*?```", "", content, flags=re.S)
        json_str = extract_first_balanced_json(cleaned)
    return {"raw": content, "json": json_str}

if __name__ == "__main__":
    def load_bus_image():
        """Load bus.jpg and encode as base64"""
        with open("bus.jpg", "rb") as img_file:
            img_data = img_file.read()
            return base64.b64encode(img_data).decode()

    image_b64 = load_bus_image()

    features: List[Feature] = [
        # image value must be raw base64 string (no "data:image/...;base64," prefix)
        {"type": "image",  "value": image_b64},
        {"type": "int",    "value": 7},
        {"type": "float",  "value": 2.5},
        {"type": "string", "value": "The capital of France is [MASK]."},
        {"type": "string", "value": "Say hi in one word."}
    ]

    res = call_ollama_qwen_vl(features)
    if res["json"]:
        try:
            parsed = json.loads(res["json"])
            print(json.dumps(parsed, indent=1))
        except Exception:
            print(res["json"])
    else:
        print(res["raw"])
    
    # --- Build detections in the same schema as the local script ---
        # --- Normalize Ollama response -> detections (robust to different shapes) ---
    if not res["json"]:
        print("[WARN] Ollama returned no JSON. Raw content follows:\n", res["raw"])
        exit(0)

    try:
        parsed = json.loads(res["json"])
    except Exception as e:
        print("[ERR] Failed to parse Ollama JSON:", e)
        print("Raw content:\n", res["raw"][:1000])
        exit(1)

    def _first_numeric_key(d: dict) -> str | None:
        keys = [k for k in d.keys() if isinstance(k, str) and k.isdigit()]
        return str(min(map(int, keys))) if keys else None

    def _extract_vision_block(obj: Any) -> dict | None:
        # A) {"index": 0, "result": {...}}
        if isinstance(obj, dict) and "result" in obj and isinstance(obj["result"], dict) and "boxes" in obj["result"]:
            return obj["result"]
        # B) {"0": {...}}
        if isinstance(obj, dict):
            k = _first_numeric_key(obj)
            if k and isinstance(obj[k], dict) and "boxes" in obj[k]:
                return obj[k]
            # C) direct {"boxes":[...]}
            if "boxes" in obj:
                return obj
        # D) list of items
        if isinstance(obj, list):
            for it in obj:
                vb = _extract_vision_block(it)
                if vb is not None:
                    return vb
        return None

    img_path = "bus.jpg"
    W, H = Image.open(img_path).convert("RGB").size

    vision = _extract_vision_block(parsed)
    if vision is None:
        print("[WARN] Could not find 'boxes' in Ollama response; saving empty detections.")
        dets = []
    else:
        raw_boxes  = vision.get("boxes", []) or []
        raw_labels = vision.get("labels", []) or []
        raw_scores = vision.get("scores", []) or []

        # sanitize boxes: must be list-like with at least 4 numbers; slice extras, fix order, clamp to image
        clean_boxes: list[list[float]] = []
        for b in raw_boxes:
            if not isinstance(b, (list, tuple)) or len(b) < 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in b[:4]]  # slice in case len>4
            if x2 < x1: x1, x2 = x2, x1
            if y2 < y1: y1, y2 = y2, y1
            x1 = max(0.0, min(x1, W - 1.0))
            x2 = max(0.0, min(x2, W - 1.0))
            y1 = max(0.0, min(y1, H - 1.0))
            y2 = max(0.0, min(y2, H - 1.0))
            clean_boxes.append([x1, y1, x2, y2])

        dets: list[dict[str, Any]] = []
        for i, box in enumerate(clean_boxes):
            label = raw_labels[i] if i < len(raw_labels) else None
            score = raw_scores[i] if i < len(raw_scores) else None
            try:
                score = float(score) if score is not None else None
            except Exception:
                score = None
            dets.append({"box": box, "label": label, "score": score})

    save_detections_json(
        image_path=img_path,
        detections=dets,
        image_size=(W, H),
        save_path="detections/online_bus.json",
        extra={"source": "multi_features_ollama_qwen2_5vl.py"}
    )
    print(f"Saved detections to detections/online_bus.json ({len(dets)} boxes)")
    if dets:
        print("First box:", dets[0])

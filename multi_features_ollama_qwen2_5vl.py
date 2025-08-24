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
            print(json.dumps(parsed, indent=2))
        except Exception:
            print(res["json"])
    else:
        print(res["raw"])


# python blip_region_captioning.py \
#   --image /path/to/bus.jpg \
#   --ann /path/to/bus_avg.json \
#   --out_dir ./outputs \
#   --label_map_json /path/to/label_map.json \  # optional

"""
BLIP region-captioning pipeline

Steps:
1) Parse annotation JSON containing bounding boxes.
2) Crop each region from the image.
3) Run BLIP to caption the full image and each crop.
4) Optionally use per-region prompts based on label/class name.
5) Overlay boxes + generated captions back onto the original image.
6) Save a result JSON with global + per-region captions.

Usage:
  python blip_region_captioning.py \
      --image /path/to/image.jpg \
      --ann /path/to/boxes.json \
      --out_dir ./outputs \
      --label_map_json /path/to/label_map.json  # optional
      --model Salesforce/blip-image-captioning-base  # optional
      --use_prompts  # optional; uses label->prompt template "A photo of a {name}."

Annotation JSON expected (flexible):
{
  "detections": [
    {"box": [x1, y1, x2, y2], "label": "18", "score": 0.98},
    ...
  ]
}
- Also supports "bbox" or dict-style {"x1":..,"y1":..,"x2":..,"y2":..}.
- If you have [x, y, w, h], pass --bbox_xywh to interpret accordingly.

Requires: pip install pillow transformers torch
"""

import os, json, argparse, math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from PIL import Image, ImageDraw, ImageFont

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


@dataclass
class Region:
    region_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    score: float = 0.0
    label: str = ""
    class_name: Optional[str] = None
    prompt: Optional[str] = None
    caption: Optional[str] = None
    crop_path: Optional[str] = None


def load_label_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    with open(path, "r") as f:
        m = json.load(f)
    # Normalize keys to strings
    return {str(k): str(v) for k, v in m.items()}


def _get_box(det: dict, bbox_xywh: bool) -> Tuple[float, float, float, float]:
    # Accept "box" or "bbox" list/tuple or x1/x2 keys
    if "box" in det:
        b = det["box"]
    elif "bbox" in det:
        b = det["bbox"]
    else:
        # try dict format
        keys = ("x1","y1","x2","y2")
        if all(k in det for k in keys):
            return float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"])
        raise ValueError("Detection does not contain 'box'/'bbox' or x1/y1/x2/y2 keys")
    if len(b) != 4:
        raise ValueError("Box must have 4 numbers")
    x, y, w_or_x2, h_or_y2 = [float(v) for v in b]
    if bbox_xywh:
        x1, y1 = x, y
        x2, y2 = x + w_or_x2, y + h_or_y2
    else:
        x1, y1, x2, y2 = x, y, w_or_x2, h_or_y2
    return x1, y1, x2, y2


def load_annotations(json_path: str, bbox_xywh: bool=False) -> List[Region]:
    with open(json_path, "r") as f:
        data = json.load(f)
    dets = data.get("detections", data)  # allow a raw list too
    regions: List[Region] = []
    for i, det in enumerate(dets, start=1):
        x1, y1, x2, y2 = _get_box(det, bbox_xywh)
        lbl = det.get("label", "")
        score = float(det.get("score", 0.0))
        regions.append(Region(
            region_id=i,
            x1=int(round(x1)),
            y1=int(round(y1)),
            x2=int(round(x2)),
            y2=int(round(y2)),
            label=str(lbl),
            score=score,
        ))
    return regions


def clamp_box(x1,y1,x2,y2,w,h):
    x1 = max(0, min(int(x1), w-1))
    y1 = max(0, min(int(y1), h-1))
    x2 = max(0, min(int(x2), w-1))
    y2 = max(0, min(int(y2), h-1))
    # ensure proper ordering
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1,y1,x2,y2


def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_blip(model_name: str, device: str):
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.to(device).eval()
    return processor, model


@torch.no_grad()
def blip_caption(processor, model, image: Image.Image, device: str, prompt: Optional[str]=None,
                 max_new_tokens: int=25, num_beams: int=5) -> str:
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
    return processor.decode(out[0], skip_special_tokens=True)


def draw_overlay(base_img: Image.Image, regions: List[Region], out_path: str):
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()

    for r in regions:
        x1,y1,x2,y2 = r.x1,r.y1,r.x2,r.y2
        draw.rectangle([x1,y1,x2,y2], outline=(255,0,0), width=3)
        # Compose overlay text
        cap = r.caption or ""
        if r.class_name: meta = f"id={r.region_id} {r.class_name} ({r.score:.2f})"
        else:            meta = f"id={r.region_id} label={r.label} ({r.score:.2f})"
        text = f"{meta}\n{cap}" if cap else meta
        # Background for legibility
        # tw, th = draw.multiline_textsize(text, font=font, spacing=2)
        # pad = 4
        # bg = [x1, max(0, y1 - th - 2*pad), x1 + tw + 2*pad, y1]
        # draw.rectangle(bg, fill=(255,0,0))
        # draw.multiline_text((x1+pad, bg[1]+pad), text, fill=(255,255,255), font=font, spacing=2)
        # Use multiline_textbbox (Pillow ≥10)
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=2)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pad = 4
        bg = [x1, max(0, y1 - th - 2*pad), x1 + tw + 2*pad, y1]
        draw.rectangle(bg, fill=(255,0,0))
        draw.multiline_text((x1+pad, bg[1]+pad), text, fill=(255,255,255), font=font, spacing=2)


    img.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to the full image")
    ap.add_argument("--ann", required=True, help="Path to annotation JSON with boxes")
    ap.add_argument("--out_dir", default="./outputs", help="Directory to save results")
    ap.add_argument("--model", default="Salesforce/blip-image-captioning-base",
                    help="BLIP model name (base or large)")
    ap.add_argument("--label_map_json", default=None,
                    help="Optional JSON mapping {\"18\":\"dog\",...}")
    ap.add_argument("--use_prompts", action="store_true",
                    help="If set, uses label->class_name to build prompts 'A photo of a {name}.'")
    ap.add_argument("--bbox_xywh", action="store_true",
                    help="Interpret boxes as [x, y, w, h] instead of [x1,y1,x2,y2]")
    ap.add_argument("--max_new_tokens", type=int, default=25)
    ap.add_argument("--num_beams", type=int, default=5)
    args = ap.parse_args()

    ensure_out_dir(args.out_dir)
    crops_dir = os.path.join(args.out_dir, "crops")
    ensure_out_dir(crops_dir)

    # Load image and annotations
    img = Image.open(args.image).convert("RGB")
    W, H = img.size
    regions = load_annotations(args.ann, bbox_xywh=args.bbox_xywh)

    # Optional label map
    label_map = load_label_map(args.label_map_json)
    for r in regions:
        if label_map:
            r.class_name = label_map.get(str(r.label), f"class_{r.label}")
        else:
            # leave None; overlay will show raw label
            pass
        if args.use_prompts:
            name = r.class_name if r.class_name else str(r.label) if r.label!="" else "object"
            r.prompt = f"A photo of a {name}."

    # Crop regions
    for r in regions:
        x1,y1,x2,y2 = clamp_box(r.x1,r.y1,r.x2,r.y2,W,H)
        crop = img.crop((x1,y1,x2,y2))
        crop_path = os.path.join(crops_dir, f"region_{r.region_id}_{r.class_name or r.label}.jpg")
        crop.save(crop_path)
        r.crop_path = crop_path

    # Load BLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    processor, model = load_blip(args.model, device)

    # Global caption
    global_caption = blip_caption(processor, model, img, device,
                                  prompt=None, max_new_tokens=args.max_new_tokens,
                                  num_beams=args.num_beams)
    print(f"[GLOBAL] {global_caption}")

    # Region captions
    for r in regions:
        crop_img = Image.open(r.crop_path).convert("RGB")
        cap = blip_caption(processor, model, crop_img, device,
                           prompt=r.prompt,
                           max_new_tokens=args.max_new_tokens,
                           num_beams=args.num_beams)
        r.caption = cap
        print(f"[REGION {r.region_id}] {cap}")

    # Save result JSON
    result = {
        "image": os.path.basename(args.image),
        "image_width": W,
        "image_height": H,
        "global_caption": global_caption,
        "regions": [
            {
                "region_id": r.region_id,
                "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2,
                "score": r.score, "label": r.label,
                "class_name": r.class_name,
                "prompt": r.prompt,
                "caption": r.caption,
                "crop_path": os.path.relpath(r.crop_path, args.out_dir) if r.crop_path else None,
            } for r in regions
        ]
    }
    result_json_path = os.path.join(args.out_dir, "blip_region_results.json")
    with open(result_json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] Wrote results to {result_json_path}")

    # Overlay on original image
    overlay_path = os.path.join(args.out_dir, "overlay_with_captions.jpg")
    draw_overlay(img, regions, overlay_path)
    print(f"[INFO] Wrote overlay image to {overlay_path}")


if __name__ == "__main__":
    main()

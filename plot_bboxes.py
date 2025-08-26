# #!/usr/bin/env python3
# import argparse
# import json
# import os
# from pathlib import Path
# from typing import Any

# from PIL import Image, ImageDraw, ImageFont


# def load_json(path: str) -> dict[str, Any]:
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)


# def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
#     """
#     Cross-version text measurement:
#     - Prefer draw.textbbox (newer Pillow)
#     - Fallback to font.getbbox
#     - Then to draw.textsize if available
#     - Last-resort heuristic
#     """
#     try:
#         left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
#         return int(right - left), int(bottom - top)
#     except Exception:
#         pass
#     try:
#         left, top, right, bottom = font.getbbox(text)
#         return int(right - left), int(bottom - top)
#     except Exception:
#         pass
#     if hasattr(draw, "textsize"):
#         w, h = draw.textsize(text, font=font)  # older Pillow
#         return int(w), int(h)
#     # Heuristic fallback
#     return max(1, 8 * len(text)), 12


# def _clip_box(box: list[int], W: int, H: int) -> list[int]:
#     x1, y1, x2, y2 = box
#     return [max(0, min(x1, W - 1)),
#             max(0, min(y1, H - 1)),
#             max(0, min(x2, W - 1)),
#             max(0, min(y2, H - 1))]


# def draw_boxes(
#     img: Image.Image,
#     detections: list[dict[str, Any]],
#     thickness: int = 3,
#     show_labels: bool = True,
#     label_map: dict[int, str] | None = None,
# ) -> Image.Image:
#     draw = ImageDraw.Draw(img)
#     try:
#         font = ImageFont.load_default()
#     except Exception:
#         font = None

#     W, H = img.size

#     for det in detections:
#         x1, y1, x2, y2 = det["box"]
#         label = det.get("label")
#         score = det.get("score")

#         # Convert to ints and clip to image bounds
#         box = _clip_box([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))], W, H)

#         # Draw rectangle with width instead of multiple outlines (cleaner & faster)
#         draw.rectangle(box, outline=(255, 0, 0), width=max(1, thickness))

#         if show_labels and font is not None and (label is not None or score is not None):
#             name = label_map.get(int(label), str(label)) if (label_map and label is not None) else str(label)
#             text = f"{name}" if score is None else f"{name} {float(score):.2f}"

#             if text:
#                 tw, th = _measure_text(draw, text, font)

#                 # Place label background above the box, clamp to image bounds
#                 bg_x0 = max(0, box[0])
#                 bg_y1 = max(0, box[1])
#                 bg_y0 = max(0, bg_y1 - th - 4)
#                 bg_x1 = min(W, bg_x0 + tw + 6)

#                 draw.rectangle([bg_x0, bg_y0, bg_x1, bg_y1], fill=(255, 0, 0))
#                 # Text with small padding
#                 tx = bg_x0 + 3
#                 ty = max(0, bg_y1 - th - 3)
#                 draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

#     return img


# def main():
#     ap = argparse.ArgumentParser(description="Plot bounding boxes from JSON onto an image.")
#     ap.add_argument("--json", required=True, help="Path to detections JSON (schema from multi-feature-updated.py).")
#     ap.add_argument("--image", default=None, help="Optional path to image; otherwise uses 'image' field in JSON.")
#     ap.add_argument("--out", default=None, help="Output image path. Default: <image_stem>_annotated.<ext>")
#     ap.add_argument("--thickness", type=int, default=3, help="Bounding box line thickness.")
#     ap.add_argument("--hide-labels", action="store_true", help="Do not render labels/scores.")
#     ap.add_argument("--label-map", default=None, help="Optional path to JSON mapping label_id->name.")
#     args = ap.parse_args()

#     payload = load_json(args.json)
#     image_path = args.image or payload.get("image")
#     if image_path is None:
#         raise SystemExit("Image path not provided and not found in JSON's 'image' field.")

#     # If a bare filename is provided, try relative to the JSON's directory
#     if not os.path.isabs(image_path):
#         candidate = Path(args.json).parent / image_path
#         image_path = str(candidate)

#     img = Image.open(image_path).convert("RGB")

#     dets = payload.get("detections", [])
#     if not isinstance(dets, list):
#         raise SystemExit("Invalid JSON: 'detections' must be a list.")

#     label_map = None
#     if args.label_map:
#         with open(args.label_map, "r", encoding="utf-8") as f:
#             label_map = {int(k): v for k, v in json.load(f).items()}

#     annotated = draw_boxes(
#         img,
#         detections=dets,
#         thickness=args.thickness,
#         show_labels=not args.hide_labels,
#         label_map=label_map
#     )

#     if args.out:
#         out_path = args.out
#     else:
#         p = Path(image_path)
#         out_path = str(p.with_name(f"{p.stem}_annotated{p.suffix}"))

#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     annotated.save(out_path)
#     print(f"Saved: {out_path}")


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    # Cross-version measurement: textbbox -> font.getbbox -> textsize -> heuristic
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return int(right - left), int(bottom - top)
    except Exception:
        pass
    try:
        left, top, right, bottom = font.getbbox(text)
        return int(right - left), int(bottom - top)
    except Exception:
        pass
    if hasattr(draw, "textsize"):
        w, h = draw.textsize(text, font=font)
        return int(w), int(h)
    return max(1, 8 * len(text)), 12


def _clip_box(box: list[int], W: int, H: int) -> list[int]:
    x1, y1, x2, y2 = box
    return [max(0, min(x1, W - 1)),
            max(0, min(y1, H - 1)),
            max(0, min(x2, W - 1)),
            max(0, min(y2, H - 1))]


def draw_boxes(
    img: Image.Image,
    detections: list[dict[str, Any]],
    thickness: int = 3,
    show_labels: bool = True,
    label_map: dict[int, str] | None = None,
) -> Image.Image:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    W, H = img.size

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det.get("label")
        score = det.get("score")

        box = _clip_box([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))], W, H)
        draw.rectangle(box, outline=(255, 0, 0), width=max(1, thickness))

        if show_labels and font is not None and (label is not None or score is not None):
            # robust label/name resolution (supports int or str labels)
            name = None
            if label_map and label is not None:
                try:
                    name = label_map.get(int(label), str(label))
                except Exception:
                    name = str(label)
            else:
                name = str(label)

            text = f"{name}" if score is None else f"{name} {float(score):.2f}"

            if text:
                tw, th = _measure_text(draw, text, font)
                bg_x0 = max(0, box[0])
                bg_y1 = max(0, box[1])
                bg_y0 = max(0, bg_y1 - th - 4)
                bg_x1 = min(W, bg_x0 + tw + 6)

                draw.rectangle([bg_x0, bg_y0, bg_x1, bg_y1], fill=(255, 0, 0))
                draw.text((bg_x0 + 3, max(0, bg_y1 - th - 3)), text, fill=(255, 255, 255), font=font)

    return img


def _resolve_image_path(json_path: str, payload: dict[str, Any], override_image: str | None) -> str:
    image_path = override_image or payload.get("image")
    if image_path is None:
        raise SystemExit(f"[{json_path}] Image path not provided and not found in JSON 'image'.")
    if not os.path.isabs(image_path):
        candidate = Path(json_path).parent / image_path
        image_path = str(candidate)
    return image_path


def _default_out_path(json_path: str, image_path: str, outdir: str | None) -> str:
    # Use JSON stem to build "<json_stem>_annotated.<img_ext>"
    # bus.json -> bus_annotated.jpg
    # online_bus.json -> online_bus_annotated.jpg
    jstem = Path(json_path).stem
    img_ext = Path(image_path).suffix or ".jpg"
    base = f"{jstem}_annotated{img_ext}"
    return str(Path(outdir) / base) if outdir else str(Path(image_path).with_name(base))


def main():
    ap = argparse.ArgumentParser(description="Plot bounding boxes from one or more JSON files onto their images.")
    ap.add_argument("--json", required=True, nargs="+", help="One or more detection JSON files.")
    ap.add_argument("--image", default=None, help="Optional image path (only valid if a single JSON is provided).")
    ap.add_argument("--out", default=None, help="Output image path (only valid for single JSON).")
    ap.add_argument("--outdir", default=None, help="Output directory when providing multiple JSONs.")
    ap.add_argument("--thickness", type=int, default=3, help="Bounding box line thickness.")
    ap.add_argument("--hide-labels", action="store_true", help="Do not render labels/scores.")
    ap.add_argument("--label-map", default=None, help="Optional path to JSON mapping label_id->name.")
    args = ap.parse_args()

    if len(args.json) > 1 and args.image:
        raise SystemExit("--image can only be used with a single --json.")
    if len(args.json) > 1 and args.out and not (args.outdir or os.path.isdir(args.out)):
        # If user passed --out with multiple JSONs, treat it as a directory if it exists
        raise SystemExit("For multiple JSONs, use --outdir to specify an output directory.")

    label_map = None
    if args.label_map:
        with open(args.label_map, "r", encoding="utf-8") as f:
            label_map = {int(k): v for k, v in json.load(f).items()}

    for jpath in args.json:
        payload = load_json(jpath)
        image_path = _resolve_image_path(jpath, payload, args.image if len(args.json) == 1 else None)

        img = Image.open(image_path).convert("RGB")
        dets = payload.get("detections", [])
        if not isinstance(dets, list):
            raise SystemExit(f"[{jpath}] Invalid JSON: 'detections' must be a list.")

        annotated = draw_boxes(
            img,
            detections=dets,
            thickness=args.thickness,
            show_labels=not args.hide_labels,
            label_map=label_map
        )

        if len(args.json) == 1 and args.out:
            out_path = args.out
        else:
            out_path = _default_out_path(jpath, image_path, args.outdir)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        annotated.save(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

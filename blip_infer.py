# python blip_infer.py --image /path/to/bus.jpg --manifest /path/to/region_manifest.json

import argparse, json, os, torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def load_model(device):
    model_name = "Salesforce/blip-image-captioning-base"  # or "large"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model

@torch.no_grad()
def caption_image(processor, model, image, device, prompt=None, max_new_tokens=25, num_beams=5):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
    return processor.decode(out[0], skip_special_tokens=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_json", default="blip_outputs.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_model(device)

    # Global caption
    full_img = Image.open(args.image).convert("RGB")
    global_caption = caption_image(processor, model, full_img, device)

    # Region captions
    with open(args.manifest, "r") as f:
        manifest = json.load(f)

    regions = []
    for r in manifest["regions"]:
        crop_path = r["crop_path"]
        # If manifest paths are relative, make them relative to its directory
        if not os.path.isabs(crop_path):
            crop_path = os.path.join(os.path.dirname(args.manifest), crop_path)
        crop = Image.open(crop_path).convert("RGB")
        # Optional: prompt the model using the label if you have semantics
        prompt = None  # e.g., f"A photo of a {r['label']}." or None
        cap = caption_image(processor, model, crop, device, prompt=prompt)
        r_out = dict(r)
        r_out["caption"] = cap
        regions.append(r_out)

    out = {
        "image": os.path.basename(args.image),
        "global_caption": global_caption,
        "regions": regions,
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

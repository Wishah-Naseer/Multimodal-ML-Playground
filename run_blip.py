# run_batch_captioning.py
import subprocess
from pathlib import Path

def main():
    for i in range(3, 21):  # loop from 1 to 20
        img = f"detections/{i}.jpg"
        ann = f"detections/{i}.json"
        out = f"./outputs/{i}"

        # ensure output dir exists (subprocess will also create inside)
        Path(out).mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", "blip_region_captioning.py",
            "--image", img,
            "--ann", ann,
            "--out_dir", out
        ]
        print(f"[INFO] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()

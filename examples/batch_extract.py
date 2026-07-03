from __future__ import annotations

from pathlib import Path

from image_to_text_vlm import ImageToTextModel


IMAGE_DIR = Path("examples/images")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

model = ImageToTextModel()

for image_path in sorted(IMAGE_DIR.glob("*")):
    if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
        continue

    text = model.run_task(image_path, task="ocr")
    output_path = OUTPUT_DIR / f"{image_path.stem}.txt"
    output_path.write_text(text, encoding="utf-8")
    print(f"Saved {output_path}")

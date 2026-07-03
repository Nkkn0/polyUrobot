# image-to-text-vlm

A clean Python starter repo for turning images into text using open Hugging Face vision-language models, plus a lightweight ASL recognition mode for robot projects.

## Two modes

### 1. Heavy image-to-text / OCR mode

Default model: `Qwen/Qwen3-VL-8B-Instruct`  
Stable fallback: `Qwen/Qwen2.5-VL-7B-Instruct`

Use this for:

- OCR from screenshots, scans, notes, receipts, and forms
- Image captioning
- Visual question answering
- Extracting structured JSON from an image

### 2. Lightweight ASL robot mode

Recommended for the PolyU robot project.

This mode uses:

- **MediaPipe Hands** to detect 21 hand landmarks in real time
- A tiny **scikit-learn MLP classifier** trained on those landmarks
- Optional serial output to send the predicted letter to a robot/controller

Why this is better for the robot:

- Runs on a normal laptop webcam
- Much lighter than Qwen/BLIP/LLaVA-style models
- Easier to stabilize across frames
- Better for static ASL alphabet signs
- Does not need a GPU once trained

There is also a lightweight pretrained Hugging Face still-image option using:

```text
abdollahhh/asl-sign-language-efficientnet-b0
```

That model is useful for demos on single images. For live robot control, use the MediaPipe landmark mode.

## Project structure

```text
image-to-text-vlm/
├── src/image_to_text_vlm/
│   ├── __init__.py
│   ├── cli.py
│   ├── vlm.py
│   ├── asl_landmarks.py
│   ├── asl_classifier.py
│   └── asl_hf.py
├── scripts/
│   ├── collect_asl_samples.py
│   ├── train_asl_landmark_model.py
│   ├── run_asl_realtime.py
│   └── asl_image_hf.py
├── examples/
│   ├── batch_extract.py
│   └── structured_json.py
├── data/
├── models/
├── tests/
│   └── test_prompts.py
├── requirements.txt
├── requirements-asl-lite.txt
├── requirements-asl-hf.txt
├── pyproject.toml
└── README.md
```

## Setup: heavy OCR / captioning mode

This is meant for a machine with a GPU. The 8B/7B models can be heavy. If you only have CPU, use the lightweight ASL mode or a hosted Hugging Face endpoint.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Optional but recommended for Qwen3-VL if your installed Transformers version does not include it yet:

```bash
pip install -U git+https://github.com/huggingface/transformers
```

If the default Qwen3 model fails on your setup, use the Qwen2.5 fallback:

```bash
export IMAGE_TEXT_MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
```

## Usage: OCR / captioning

### OCR

```bash
image-to-text path/to/image.png --task ocr
```

### Caption an image

```bash
image-to-text path/to/image.jpg --task caption
```

### Ask a custom question

```bash
image-to-text path/to/screenshot.png --prompt "What are the main numbers in this chart?"
```

### Extract JSON

```bash
image-to-text receipt.jpg --task json
```

## Setup: lightweight ASL robot mode

Install only the lightweight dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-asl-lite.txt
pip install -e .
```

## Collect your own ASL samples

You should collect samples in the same environment where the robot demo will happen. Same camera, same lighting, same background if possible.

For each letter, run:

```bash
python scripts/collect_asl_samples.py --label A --max-samples 100 --mirror
python scripts/collect_asl_samples.py --label B --max-samples 100 --mirror
python scripts/collect_asl_samples.py --label C --max-samples 100 --mirror
```

Continue for the letters you need.

During collection:

- Press **space** to save one sample
- Press **q** to quit
- Start with 5-10 letters first before doing the full alphabet
- Collect different hand positions, distances, and slight rotations

Note: ASL letters **J** and **Z** are dynamic signs, so a static image classifier will not capture them properly. For a school robot demo, start with static letters like A, B, C, D, E, F, I, L, O, V, W, Y.

## Train the tiny classifier

After collecting samples:

```bash
python scripts/train_asl_landmark_model.py \
  --csv data/asl_landmarks.csv \
  --output models/asl_landmark_mlp.joblib
```

## Run real-time ASL recognition

```bash
python scripts/run_asl_realtime.py \
  --model models/asl_landmark_mlp.joblib \
  --mirror \
  --min-confidence 0.75 \
  --stable-frames 7
```

This will show the predicted letter on the webcam feed. A letter is accepted only after it is stable for several frames.

## Send the predicted letter to the robot

If the robot/controller listens on a serial port, use:

```bash
python scripts/run_asl_realtime.py \
  --model models/asl_landmark_mlp.joblib \
  --mirror \
  --serial-port /dev/ttyUSB0 \
  --baud 115200
```

On macOS, the serial port may look like:

```bash
/dev/cu.usbserial-XXXX
```

The script sends one letter followed by a newline, for example:

```text
A
```

Your robot code can map that letter to the mDrawBot writing routine.

## Lightweight pretrained ASL image classifier

Use this first for a quick prototype before collecting your own training data.

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-asl-hf.txt
pip install -e .
```

Test one still image:

```bash
python scripts/asl_image_hf.py path/to/asl_letter.jpg --top-k 5
```

Test live with your webcam:

```bash
python scripts/run_asl_hf_webcam.py \
  --mirror \
  --min-confidence 0.70 \
  --stable-frames 7
```

During the webcam test:

- Put your hand inside the center box
- Use a plain background and good lighting
- Press **s** to save the cropped hand image
- Press **q** to quit

This uses:

```text
abdollahhh/asl-sign-language-efficientnet-b0
```

The model is much smaller than Qwen and is fine for a quick demo, but it was trained on controlled images. If it struggles with your webcam/background, use the MediaPipe landmark path above for the final robot demo.

Static image classifiers also do not properly handle ASL **J** and **Z**, because those letters are motion-based.

## Python API: OCR mode

```python
from image_to_text_vlm import ImageToTextModel

model = ImageToTextModel()
text = model.generate(
    image_path="examples/sample.png",
    prompt="Extract all visible text exactly. Preserve line breaks.",
)
print(text)
```

## Python API: ASL image mode

```python
from image_to_text_vlm import ASLImageClassifier

classifier = ASLImageClassifier()
label, confidence = classifier.predict("hand_sign.jpg")
print(label, confidence)
```

## Environment variables

Copy `.env.example` to `.env` or export these manually:

```bash
IMAGE_TEXT_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
IMAGE_TEXT_MAX_NEW_TOKENS=1024
```

## Practical robot pipeline

The final technical flow should be:

```text
webcam frame
  -> MediaPipe hand landmarks
  -> normalized 63-feature vector
  -> tiny ASL classifier
  -> stable predicted letter
  -> robot command: write that letter
  -> mDrawBot letter path routine
```

This is better than sending the whole camera frame into a giant image-to-text model every time.

## Notes

- For exact OCR, ask the VLM to preserve line breaks and not infer missing text.
- For receipts/forms, ask for JSON and validate the output downstream.
- For production OCR, keep human review for critical documents because VLMs can still hallucinate.
- For robot ASL mode, add a confidence threshold and require the same prediction for several frames before moving the robot.

## Push to GitHub

After extracting this repo:

```bash
git init
git add .
git commit -m "Add image-to-text and lightweight ASL robot modes"
git branch -M main
git remote add origin https://github.com/Nkkn0/polyUrobot.git
git push -u origin main
```

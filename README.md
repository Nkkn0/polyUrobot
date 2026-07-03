# polyUrobot Image-to-Text / ASL / Handwriting

This repository contains Python demos for the PolyU robot project. The goal is to recognize letters from different visual inputs and later connect the predicted letter to the robot so it can write it.

## Main Features

### 1. ASL Letter Recognition

Show an ASL alphabet sign to the webcam. The model predicts the letter.

### 2. Handwritten Letter Recognition

Upload or pass an image of a handwritten letter. The model predicts the letter.
---

# Quick Start

## 0. Create and Activate the Python Environment

Run this once from the repo folder:

```bash
cd /Users/nossaibakheiri/Downloads/image-to-text-vlm

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
```

If the environment already exists, just activate it:

```bash
cd /Users/nossaibakheiri/Downloads/image-to-text-vlm
source .venv/bin/activate
```

---

# 1. ASL Recognition Mode

This mode predicts an ASL alphabet letter from your webcam.

It uses a pretrained lightweight ASL image classifier.

## Install ASL Dependencies

```bash
pip install -r requirements-asl-hf.txt
pip install -e .
```

## Run ASL Webcam Recognition

```bash
python scripts/run_asl_hf_webcam.py --camera 0 --mirror --min-confidence 0.70 --stable-frames 7
```

When the camera window opens:

```text
Show one ASL letter sign inside the center box.
Press q to quit.
Press s to save a crop for testing.
```

Avoid these at first:

```text
J, Z
```

These require motion, not just a single static image.

## Test ASL on One Image

```bash
python scripts/asl_image_hf.py /path/to/asl_letter.jpg --top-k 5
```

Example:

```bash
python scripts/asl_image_hf.py /Users/nossaibakheiri/Downloads/asl_A.jpg --top-k 5
```

---

# 2. Handwritten Letter Recognition Mode

This mode predicts a handwritten letter from an uploaded image.

Current model:

```text
microsoft/trocr-base-handwritten
```

This model works better for real handwriting than the earlier EMNIST classifier. However, for the final robot demo, the most reliable solution will still be to train a small model on your own handwritten-letter images.

## Install Handwriting Dependencies

```bash
pip install -r requirements-handwriting.txt
pip install -e .
```

## Run the Upload App

```bash
streamlit run scripts/handwriting_upload_app.py
```

Then open the local Streamlit link in your browser and upload an image.


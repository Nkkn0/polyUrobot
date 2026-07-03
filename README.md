# polyUrobot Image-to-Text / ASL / Handwriting

## How to run the project

This repo has two main demo modes:

1. **ASL letter recognition** — show an ASL alphabet sign to the webcam.
2. **Handwritten letter recognition** — upload or pass an image of a handwritten letter and get the predicted letter.

---

## 0. Create and activate the Python environment

Run this once from the repo folder:

```bash
cd /Users/nossaibakheiri/Downloads/image-to-text-vlm

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
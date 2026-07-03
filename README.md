polyUrobot Image-to-Text / ASL / Handwriting

This repo contains Python demos for the PolyU robot project:

ASL letter recognition — show an ASL alphabet sign to the webcam and predict the letter.
Handwritten letter recognition — upload or pass an image of a handwritten letter and predict the letter.
General image-to-text / OCR — use a vision-language model to describe or extract text from images.
Quick Start
0. Create and activate the Python environment

Run this once from the repo folder:

cd /Users/nossaibakheiri/Downloads/image-to-text-vlm

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e .

If the environment already exists, just activate it:

cd /Users/nossaibakheiri/Downloads/image-to-text-vlm
source .venv/bin/activate
1. ASL Recognition Mode

This mode predicts an ASL alphabet letter from your webcam.

It uses a pretrained lightweight ASL image classifier.

Install ASL dependencies
pip install -r requirements-asl-hf.txt
pip install -e .
Run ASL webcam recognition
python scripts/run_asl_hf_webcam.py --camera 0 --mirror --min-confidence 0.70 --stable-frames 7

When the camera window opens:

Show one ASL letter sign inside the center box.
Press q to quit.
Press s to save a crop for testing.

Start with easy static letters:

A, B, C, D, E, F, I, L, O, V, W, Y

Avoid these at first:

J, Z

because they require motion, not just a single static image.

Test ASL on one image
python scripts/asl_image_hf.py /path/to/asl_letter.jpg --top-k 5

Example:

python scripts/asl_image_hf.py /Users/nossaibakheiri/Downloads/asl_A.jpg --top-k 5
2. Handwritten Letter Recognition Mode

This mode predicts a handwritten letter from an uploaded image.

Current model:

microsoft/trocr-base-handwritten

This model is better for real handwriting than the earlier EMNIST classifier, but for the final robot demo, the most reliable version will still be training a small model on your own handwriting images.

Install handwriting dependencies
pip install -r requirements-handwriting.txt
pip install -e .
Run the upload app
streamlit run scripts/handwriting_upload_app.py

Then open the local Streamlit link in your browser and upload an image.

Best input:

One large capital handwritten letter
Dark pen or marker
White paper
Good lighting
Not a whole word
Test handwriting from terminal
python scripts/predict_handwritten_letter.py /path/to/letter.jpg --top-k 5
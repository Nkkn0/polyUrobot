from PIL import Image, ImageDraw

from image_to_text_vlm.handwriting_hf import label_to_letter, preprocess_handwritten_letter


def test_label_to_letter_variants():
    assert label_to_letter("A") == "A"
    assert label_to_letter("LABEL_0") == "A"
    assert label_to_letter("LABEL_25") == "Z"
    assert label_to_letter("25") == "Z"


def test_preprocess_handwritten_letter_returns_rgb_square():
    img = Image.new("RGB", (200, 120), "white")
    draw = ImageDraw.Draw(img)
    draw.text((80, 40), "A", fill="black")
    out = preprocess_handwritten_letter(img)
    assert out.mode == "RGB"
    assert out.size[0] == out.size[1]

from __future__ import annotations

from image_to_text_vlm import ImageToTextModel

model = ImageToTextModel()

result = model.generate(
    image_path="examples/receipt.jpg",
    prompt=(
        "Extract receipt information as valid JSON only with fields: "
        "merchant, date, currency, subtotal, tax, total, and items. "
        "Each item should have name, quantity, unit_price, and line_total. "
        "Use null when a field is not visible."
    ),
)

print(result)

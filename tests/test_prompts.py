from image_to_text_vlm.vlm import TASK_PROMPTS


def test_required_task_prompts_exist():
    assert "ocr" in TASK_PROMPTS
    assert "caption" in TASK_PROMPTS
    assert "json" in TASK_PROMPTS


def test_ocr_prompt_avoids_inference():
    assert "Do not infer" in TASK_PROMPTS["ocr"]

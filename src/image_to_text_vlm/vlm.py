from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
FALLBACK_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


TASK_PROMPTS = {
    "ocr": (
        "Extract all visible text from this image exactly. "
        "Preserve line breaks and layout as much as possible. "
        "Do not add explanations. Do not infer missing text."
    ),
    "caption": "Describe this image clearly and concisely.",
    "json": (
        "Extract the important information from this image as valid JSON only. "
        "Use null for missing fields. Do not include markdown."
    ),
}


@dataclass
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.0
    do_sample: bool = False


def _load_dotenv_if_available() -> None:
    """Load .env when python-dotenv is installed; keep ASL-only installs lightweight."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _get_model_class(model_id: str):
    """Return the right Transformers class for the selected Qwen VLM."""
    normalized = model_id.lower()

    if "qwen3-vl" in normalized:
        try:
            from transformers import Qwen3VLForConditionalGeneration

            return Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen3-VL is not available in your installed Transformers version. "
                "Run: pip install -U git+https://github.com/huggingface/transformers "
                f"or use fallback model {FALLBACK_MODEL_ID}."
            ) from exc

    if "qwen2.5-vl" in normalized or "qwen2_5-vl" in normalized:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration

            return Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen2.5-VL is not available in your installed Transformers version. "
                "Run: pip install -U transformers."
            ) from exc

    raise ValueError(
        "This repo is optimized for Qwen3-VL and Qwen2.5-VL models. "
        "Set IMAGE_TEXT_MODEL_ID to Qwen/Qwen3-VL-8B-Instruct or "
        "Qwen/Qwen2.5-VL-7B-Instruct."
    )


class ImageToTextModel:
    """Small wrapper around Qwen VLM image-to-text generation."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        device_map: str = "auto",
        dtype: str = "auto",
    ) -> None:
        _load_dotenv_if_available()
        self.model_id = model_id or os.getenv("IMAGE_TEXT_MODEL_ID", DEFAULT_MODEL_ID)
        self.device_map = device_map
        self.dtype = dtype

        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Transformers is required for OCR/captioning mode. "
                "Install it with: pip install -r requirements.txt"
            ) from exc

        model_cls = _get_model_class(self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = model_cls.from_pretrained(
            self.model_id,
            dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()

    def generate(
        self,
        image_path: str | Path,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Generate text from a single image and prompt."""
        try:
            import torch
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError(
                "Qwen VLM generation needs torch and qwen-vl-utils. "
                "Install OCR/captioning dependencies with: pip install -r requirements.txt"
            ) from exc

        image_path = Path(image_path).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        config = config or GenerationConfig(
            max_new_tokens=int(os.getenv("IMAGE_TEXT_MAX_NEW_TOKENS", "1024"))
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
        }
        if config.do_sample:
            generation_kwargs["temperature"] = config.temperature

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        # Remove prompt tokens from output.
        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text.strip()

    def run_task(self, image_path: str | Path, task: str = "ocr") -> str:
        """Run a named task: ocr, caption, or json."""
        if task not in TASK_PROMPTS:
            valid = ", ".join(sorted(TASK_PROMPTS))
            raise ValueError(f"Unknown task '{task}'. Valid tasks: {valid}")
        return self.generate(image_path=image_path, prompt=TASK_PROMPTS[task])

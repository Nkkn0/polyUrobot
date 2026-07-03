from __future__ import annotations

from pathlib import Path

import click

from .vlm import GenerationConfig, ImageToTextModel, TASK_PROMPTS


@click.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--task",
    type=click.Choice(sorted(TASK_PROMPTS.keys())),
    default="ocr",
    show_default=True,
    help="Prebuilt task prompt.",
)
@click.option(
    "--prompt",
    type=str,
    default=None,
    help="Custom prompt. Overrides --task when provided.",
)
@click.option(
    "--model-id",
    type=str,
    default=None,
    help="Hugging Face model id. Overrides IMAGE_TEXT_MODEL_ID.",
)
@click.option(
    "--max-new-tokens",
    type=int,
    default=1024,
    show_default=True,
    help="Maximum number of tokens to generate.",
)
def main(
    image_path: Path,
    task: str,
    prompt: str | None,
    model_id: str | None,
    max_new_tokens: int,
) -> None:
    """Convert IMAGE_PATH to text with a Qwen vision-language model."""
    selected_prompt = prompt or TASK_PROMPTS[task]
    model = ImageToTextModel(model_id=model_id)
    result = model.generate(
        image_path=image_path,
        prompt=selected_prompt,
        config=GenerationConfig(max_new_tokens=max_new_tokens),
    )
    click.echo(result)


if __name__ == "__main__":
    main()

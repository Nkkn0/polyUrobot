"""Image-to-text and lightweight ASL robot utilities."""

__all__ = ["ImageToTextModel", "ASLLandmarkClassifier", "ASLImageClassifier"]


def __getattr__(name: str):
    """Lazy exports so ASL-lite can run without installing heavy VLM dependencies."""
    if name == "ImageToTextModel":
        from .vlm import ImageToTextModel

        return ImageToTextModel
    if name == "ASLLandmarkClassifier":
        from .asl_classifier import ASLLandmarkClassifier

        return ASLLandmarkClassifier
    if name == "ASLImageClassifier":
        from .asl_hf import ASLImageClassifier

        return ASLImageClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

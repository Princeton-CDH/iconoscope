"""CLI command handlers loaded on demand."""

from iconoscope.dataset import ImageDataset
from iconoscope.models import AVAILABLE_MODELS, EmbeddingModel


def get_model(name: str) -> EmbeddingModel:
    """Look up an embedding model by name; exit with an error if unknown."""
    try:
        return AVAILABLE_MODELS[name]
    except KeyError:
        raise SystemExit(f"Unknown model `{name}`") from None


def require_features(img_dataset: ImageDataset, model: EmbeddingModel) -> None:
    """Exit with an error if the dataset has no features for the specified model."""
    if not img_dataset.has_features(model):
        raise SystemExit(
            f"No features for `{model.name}`; run `iconoscope embed --model {model.name}` first"
        )

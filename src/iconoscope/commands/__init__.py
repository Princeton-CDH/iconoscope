"""CLI command handlers loaded on demand."""

import argparse

from iconoscope.dataset import ImageDataset
from iconoscope.models import AVAILABLE_MODELS, EmbeddingConfig, EmbeddingModel


def get_model(name: str) -> EmbeddingModel:
    """Look up an embedding model by name; exit with an error if unknown."""
    try:
        return AVAILABLE_MODELS[name]
    except KeyError:
        raise SystemExit(f"Unknown model `{name}`") from None


def get_embedding_config(args: argparse.Namespace) -> EmbeddingConfig:
    """Build an embedding configuration (model + preprocessing options) from CLI args."""
    model = get_model(args.model)
    opts = {field: getattr(args, field) for field in EmbeddingConfig.option_names()}
    return EmbeddingConfig(model, **opts)


def require_features(img_dataset: ImageDataset, config: EmbeddingConfig) -> None:
    """Exit with an error if the dataset has no features for the specified
    model and preprocessing configuration."""
    if not img_dataset.has_features(config):
        raise SystemExit(
            f"No features for `{config.model.name}/{config.key}`; "
            f"run `iconoscope embed --model {config.model.name}` "
            "with the same preprocessing options first"
        )

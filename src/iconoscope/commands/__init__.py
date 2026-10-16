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


def get_preprocess_opts(args: argparse.Namespace) -> dict:
    opts = {}
    if getattr(args, "center_crop", None):
        opts["center_crop"] = args.center_crop
    if getattr(args, "grayscale", None):
        opts["grayscale"] = args.grayscale
    return opts


def get_embedding_config(args) -> EmbeddingConfig:
    model = get_model(args.model)
    opts = {field: getattr(args, field) for field in EmbeddingConfig.option_names()}
    return EmbeddingConfig(model, **opts)


def require_features(
    img_dataset: ImageDataset, model: EmbeddingModel, **preprocssing_opts
) -> None:
    """Exit with an error if the dataset has no features for the specified model."""
    if not img_dataset.has_features(model, **preprocssing_opts):
        raise SystemExit(
            f"No features for `{model.name}`; run `iconoscope embed --model {model.name}` first"
        )

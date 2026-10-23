"""CLI command handlers loaded on demand."""

import argparse

from iconoscope.dataset import ImageDataset
from iconoscope.models import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    EmbeddingConfig,
    EmbeddingModel,
)


def get_model(name: str) -> EmbeddingModel:
    """Look up an embedding model by name; exit with an error if unknown."""
    try:
        return AVAILABLE_MODELS[name]
    except KeyError:
        raise SystemExit(f"Unknown model `{name}`") from None


def get_embedding_config(args: argparse.Namespace) -> EmbeddingConfig:
    """Build an embedding configuration (model + preprocessing options) from CLI args."""
    model = get_model(args.model or DEFAULT_MODEL.name)
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


def get_stored_config(
    img_dataset: ImageDataset, label: str, available: dict | None = None
) -> EmbeddingConfig:
    """Look up a stored embedding configuration by ``model/key`` label;
    exit with an error listing available configurations if not found."""
    if available is None:
        available = img_dataset.configs_by_label()
    try:
        return available[label]
    except KeyError:
        options = ", ".join(available) or "none; run `iconoscope embed` first"
        raise SystemExit(
            f"No features for `{label}`; available configurations: {options}"
        ) from None


def resolve_config(
    img_dataset: ImageDataset, args: argparse.Namespace
) -> EmbeddingConfig:
    """Determine a single embedding configuration from CLI args, specified
    either as ``--config MODEL/KEY`` or as model + preprocessing options;
    exits with an error if the dataset has no features for it."""
    if getattr(args, "config", None):
        return get_stored_config(img_dataset, args.config)
    config = get_embedding_config(args)
    require_features(img_dataset, config)
    return config


def resolve_configs(
    img_dataset: ImageDataset, args: argparse.Namespace
) -> list[EmbeddingConfig]:
    """Determine embedding configurations from CLI args: one or more
    ``--config MODEL/KEY`` labels, model + preprocessing options, or (if
    nothing is specified) all configurations stored in the dataset."""
    if args.config:
        available = img_dataset.configs_by_label()
        # dict to drop duplicates while preserving order
        labels = dict.fromkeys(args.config)
        return [get_stored_config(img_dataset, lbl, available) for lbl in labels]
    if args.model is not None or args.center_crop is not None or args.grayscale:
        config = get_embedding_config(args)
        require_features(img_dataset, config)
        return [config]
    configs = list(img_dataset.configs_by_label().values())
    if not configs:
        raise SystemExit("No features in dataset; run `iconoscope embed` first")
    return configs

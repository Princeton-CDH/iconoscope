"""CLI command handlers loaded on demand."""

import argparse

from iconoscope.dataset import ImageDataset
from iconoscope.models import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    EmbeddingConfig,
    EmbeddingModel,
)


def load_dataset(path) -> ImageDataset:
    """Open an existing image dataset; exit with an error message if it does
    not exist or is not a valid iconoscope dataset."""
    try:
        return ImageDataset(storage_path=path)
    except (ValueError, FileNotFoundError) as err:
        # InvalidDatasetError is a ValueError
        raise SystemExit(f"Error: {err}") from None


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


def model_opts_specified(args: argparse.Namespace) -> bool:
    """True if a model or any preprocessing option was specified."""
    return args.model is not None or any(
        getattr(args, opt) not in (None, False)
        for opt in EmbeddingConfig.option_names()
    )


def default_stored_config(img_dataset: ImageDataset) -> EmbeddingConfig:
    """Embedding configuration to use when none is specified: the only stored
    configuration if there is just one; otherwise the default model with
    default preprocessing, if stored. Exits with an error if there are no
    stored features or the choice is ambiguous."""
    available = img_dataset.configs_by_label()
    if not available:
        raise SystemExit("No features in dataset; run `iconoscope embed` first")
    if len(available) == 1:
        return next(iter(available.values()))
    default = EmbeddingConfig(DEFAULT_MODEL)
    if default.label in available:
        return default
    raise SystemExit(
        "Dataset has features for multiple configurations; specify one with "
        f"--config or --model: {', '.join(available)}"
    )


def resolve_config(
    img_dataset: ImageDataset, args: argparse.Namespace
) -> EmbeddingConfig:
    """Determine a single embedding configuration from CLI args, specified
    either as ``--config MODEL/KEY`` or as model + preprocessing options;
    exits with an error if the dataset has no features for it. If neither
    is specified, uses :func:`default_stored_config`."""
    if getattr(args, "config", None):
        return get_stored_config(img_dataset, args.config)
    if not model_opts_specified(args):
        return default_stored_config(img_dataset)
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
    if model_opts_specified(args):
        return [resolve_config(img_dataset, args)]
    configs = list(img_dataset.configs_by_label().values())
    if not configs:
        raise SystemExit("No features in dataset; run `iconoscope embed` first")
    return configs

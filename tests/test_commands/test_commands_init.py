import argparse
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from iconoscope import cli
from iconoscope.commands import get_embedding_config
from iconoscope.models import AVAILABLE_MODELS, DEFAULT_MODEL, EmbeddingConfig


def test_get_embedding_config_defaults():
    args = argparse.Namespace(
        model=DEFAULT_MODEL.name, center_crop=None, grayscale=False
    )
    assert get_embedding_config(args) == EmbeddingConfig(DEFAULT_MODEL)


def test_get_embedding_config_opts():
    model = AVAILABLE_MODELS["clip"]
    args = argparse.Namespace(model="clip", center_crop=0.6, grayscale=True)
    config = get_embedding_config(args)
    assert config == EmbeddingConfig(model, center_crop=0.6, grayscale=True)


def test_get_embedding_config_unknown_model():
    args = argparse.Namespace(model="bogus", center_crop=None, grayscale=False)
    with pytest.raises(SystemExit, match="Unknown model `bogus`"):
        get_embedding_config(args)


def parse_cli_args(argv: list[str]) -> argparse.Namespace:
    """Run the CLI parser and return the args passed to the command handler."""
    handler = MagicMock()
    command_module = ModuleType("command")
    command_module.main = handler
    with (
        patch("iconoscope.cli.import_module", return_value=command_module),
        patch("sys.argv", ["iconoscope", *argv]),
    ):
        cli.main()
    return handler.call_args.args[0]


def test_get_embedding_config_no_model():
    # model is None when not specified for commands after embed
    args = argparse.Namespace(model=None, center_crop=None, grayscale=True)
    assert get_embedding_config(args) == EmbeddingConfig(DEFAULT_MODEL, grayscale=True)


@pytest.mark.parametrize("command", ["embed", "mosaic", "cluster"])
def test_cli_args_in_sync_with_embedding_config(command, tmp_path):
    # every model-based command must provide all EmbeddingConfig options,
    # so get_embedding_config works for any of them
    dataset = tmp_path / "data.h5"
    dataset.touch()
    argv = [command, str(dataset)]
    if command == "cluster":
        argv.append("5")
    args = parse_cli_args(argv)
    for option in EmbeddingConfig.option_names():
        assert hasattr(args, option), f"`{command}` is missing --{option}"
    assert get_embedding_config(args) == EmbeddingConfig(DEFAULT_MODEL)


def test_cli_args_to_embedding_config(tmp_path):
    dataset = tmp_path / "data.h5"
    dataset.touch()
    args = parse_cli_args(
        ["embed", str(dataset), "--model", "clip", "--center-crop", "60", "--grayscale"]
    )
    config = get_embedding_config(args)
    assert config.model == AVAILABLE_MODELS["clip"]
    assert config.center_crop == 0.6
    assert config.grayscale is True
    assert config.key == "center_crop=0.6_grayscale"


def test_require_features(tmp_image_dir):
    from iconoscope.commands import require_features
    from iconoscope.dataset import ImageDataset

    dataset = ImageDataset.create(tmp_image_dir, tmp_image_dir / "data.h5")
    config = EmbeddingConfig(DEFAULT_MODEL, grayscale=True)
    with pytest.raises(
        SystemExit, match=f"No features for `{DEFAULT_MODEL.name}/grayscale`"
    ):
        require_features(dataset, config)

    df = dataset.get_image_data().select("image_path")
    df = df.with_columns(features=np.zeros((df.height, 4)))
    dataset.save_features(df, config)
    # no error when features for the configuration are present
    require_features(dataset, config)
    # still an error for a different configuration of the same model
    with pytest.raises(SystemExit):
        require_features(dataset, EmbeddingConfig(DEFAULT_MODEL))


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["mosaic"], None),
        (["cluster", "5"], None),
        (["mosaic", "--config", "clip/grayscale"], "clip/grayscale"),
        (["cluster", "5", "-c", "clip/grayscale"], "clip/grayscale"),
    ],
)
def test_cli_stored_config_single(argv, expected, tmp_path):
    dataset = tmp_path / "data.h5"
    dataset.touch()
    args = parse_cli_args([argv[0], str(dataset), *argv[1:]])
    assert args.config == expected
    # no model default from the parser; determined by the command from the dataset
    assert args.model is None


def test_cli_umap_args(tmp_path):
    dataset = tmp_path / "data.h5"
    dataset.touch()
    # nothing specified: no model, no config (command uses all configs)
    args = parse_cli_args(["umap", str(dataset)])
    assert args.model is None
    assert args.config is None
    assert args.update is False
    # multiple configs allowed
    args = parse_cli_args(
        ["umap", str(dataset), "-c", "clip/defaults", "--config", "dinov2/grayscale"]
    )
    assert args.config == ["clip/defaults", "dinov2/grayscale"]
    # model options
    args = parse_cli_args(
        ["umap", str(dataset), "--model", "clip", "--grayscale", "-u"]
    )
    assert get_embedding_config(args) == EmbeddingConfig(
        AVAILABLE_MODELS["clip"], grayscale=True
    )
    assert args.update is True


@pytest.mark.parametrize("command", ["mosaic", "cluster", "umap"])
@pytest.mark.parametrize("opt", [["--model", "clip"], ["--grayscale"]])
def test_cli_config_excludes_model_options(command, opt, tmp_path, capsys):
    dataset = tmp_path / "data.h5"
    dataset.touch()
    argv = [command, str(dataset), "--config", "clip/defaults", *opt]
    if command == "cluster":
        argv.insert(2, "5")
    with pytest.raises(SystemExit):
        parse_cli_args(argv)
    assert "--config cannot be combined" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["mosaic", "cluster", "umap"])
def test_cli_requires_existing_dataset(command, tmp_path, capsys):
    argv = [command, str(tmp_path / "missing.h5")]
    if command == "cluster":
        argv.append("5")
    with pytest.raises(SystemExit):
        parse_cli_args(argv)
    assert "is not an existing dataset file" in capsys.readouterr().err


def test_cli_embed_has_no_config(tmp_path):
    dataset = tmp_path / "data.h5"
    dataset.touch()
    with pytest.raises(SystemExit):
        parse_cli_args(["embed", str(dataset), "--config", "clip/defaults"])


def save_test_features(dataset, *configs):
    df = dataset.get_image_data().select("image_path")
    for config in configs:
        dataset.save_features(
            df.with_columns(features=np.zeros((df.height, 4))), config
        )


def no_model_args(**opts) -> argparse.Namespace:
    defaults = {"model": None, "center_crop": None, "grayscale": False, "config": None}
    return argparse.Namespace(**(defaults | opts))


def test_default_stored_config_single(tmp_image_dir):
    from iconoscope.commands import default_stored_config, resolve_config
    from iconoscope.dataset import ImageDataset

    dataset = ImageDataset.create(tmp_image_dir, tmp_image_dir / "data.h5")
    clip_gray = EmbeddingConfig(AVAILABLE_MODELS["clip"], grayscale=True)
    save_test_features(dataset, clip_gray)
    # only one stored config: used as default, even if not the default model
    assert default_stored_config(dataset) == clip_gray
    assert resolve_config(dataset, no_model_args()) == clip_gray


def test_default_stored_config_multiple_with_default(tmp_image_dir):
    from iconoscope.commands import default_stored_config
    from iconoscope.dataset import ImageDataset

    dataset = ImageDataset.create(tmp_image_dir, tmp_image_dir / "data.h5")
    save_test_features(
        dataset,
        EmbeddingConfig(AVAILABLE_MODELS["clip"]),
        EmbeddingConfig(DEFAULT_MODEL),
    )
    # multiple stored configs: default model with default options is used
    assert default_stored_config(dataset) == EmbeddingConfig(DEFAULT_MODEL)


def test_default_stored_config_ambiguous(tmp_image_dir):
    from iconoscope.commands import default_stored_config
    from iconoscope.dataset import ImageDataset

    dataset = ImageDataset.create(tmp_image_dir, tmp_image_dir / "data.h5")
    save_test_features(
        dataset,
        EmbeddingConfig(AVAILABLE_MODELS["clip"]),
        EmbeddingConfig(DEFAULT_MODEL, grayscale=True),
    )
    with pytest.raises(SystemExit, match="multiple configurations.*clip/defaults"):
        default_stored_config(dataset)


def test_default_stored_config_no_features(tmp_image_dir):
    from iconoscope.commands import default_stored_config
    from iconoscope.dataset import ImageDataset

    dataset = ImageDataset.create(tmp_image_dir, tmp_image_dir / "data.h5")
    with pytest.raises(SystemExit, match="No features in dataset"):
        default_stored_config(dataset)


def test_resolve_config_model_opts_not_defaulted(tmp_image_dir):
    # if model options are specified, the single stored config is not used
    from iconoscope.commands import resolve_config
    from iconoscope.dataset import ImageDataset

    dataset = ImageDataset.create(tmp_image_dir, tmp_image_dir / "data.h5")
    save_test_features(dataset, EmbeddingConfig(AVAILABLE_MODELS["clip"]))
    with pytest.raises(
        SystemExit, match=f"No features for `{DEFAULT_MODEL.name}/grayscale`"
    ):
        resolve_config(dataset, no_model_args(grayscale=True))


def test_cli_center_crop_100(tmp_path):
    # 100% keeps the whole image (without model cropping); not the default
    dataset = tmp_path / "data.h5"
    dataset.touch()
    args = parse_cli_args(["embed", str(dataset), "--center-crop", "100"])
    assert get_embedding_config(args) == EmbeddingConfig(DEFAULT_MODEL, center_crop=1.0)


@pytest.mark.parametrize("command", ["embed", "info", "mosaic", "cluster", "umap"])
def test_command_invalid_dataset(command, tmp_path):
    from importlib import import_module

    # existing file that is not an iconoscope dataset
    dataset = tmp_path / "data.h5"
    dataset.write_text("not a dataset")
    args = argparse.Namespace(
        dataset=dataset,
        model=None,
        config=None,
        center_crop=None,
        grayscale=False,
        update=False,
    )
    main = import_module(f"iconoscope.commands.{command}").main
    with pytest.raises(SystemExit, match="Error: .* is not an HDF5 file"):
        main(args)


def test_load_dataset(tmp_image_dir, tmp_path):
    from iconoscope.commands import load_dataset
    from iconoscope.dataset import ImageDataset

    ImageDataset.create(tmp_image_dir, tmp_path / "data.h5")
    assert isinstance(load_dataset(tmp_path / "data.h5"), ImageDataset)
    with pytest.raises(SystemExit, match="does not exist"):
        load_dataset(tmp_path / "missing.h5")

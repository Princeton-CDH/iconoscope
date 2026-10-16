import argparse
from types import ModuleType
from unittest.mock import MagicMock, patch

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
    args = argparse.Namespace(model="clip", center_crop=60.0, grayscale=True)
    config = get_embedding_config(args)
    assert config == EmbeddingConfig(model, center_crop=60.0, grayscale=True)


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
    assert config.center_crop == 60.0
    assert config.grayscale is True
    assert config.key == "center_crop=60.0_grayscale"

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from iconoscope import cli

logger = logging.getLogger(__name__)
MAX_CLI_IMPORT_SECONDS = 1.0


def test_embed_args(tmp_path: Path):
    # Test parser arguments without importing the embedding stack.
    dataset = tmp_path / "data.h5"
    dataset.touch()
    handler = MagicMock()
    command_module = ModuleType("iconoscope.commands.embed")
    command_module.main = handler
    with (
        patch("iconoscope.cli.import_module", return_value=command_module) as importer,
        patch("sys.argv", ["iconoscope", "embed", str(dataset)]),
    ):
        cli.main()

    importer.assert_called_once_with("iconoscope.commands.embed")
    args = handler.call_args.args[0]
    assert args.dataset == dataset


def test_embed_missing_dataset_does_not_import_handler(tmp_path: Path):
    with (
        patch("iconoscope.cli.import_module") as importer,
        patch("sys.argv", ["iconoscope", "embed", str(tmp_path / "missing.h5")]),
        pytest.raises(SystemExit),
    ):
        cli.main()

    importer.assert_not_called()


def test_create_args(tmp_path: Path):
    out = tmp_path / "data.h5"
    handler = MagicMock()
    command_module = ModuleType("iconoscope.commands.create")
    command_module.main = handler
    with (
        patch("iconoscope.cli.import_module", return_value=command_module) as importer,
        patch("sys.argv", ["iconoscope", "create", str(tmp_path), str(out)]),
    ):
        cli.main()

    importer.assert_called_once_with("iconoscope.commands.create")
    args = handler.call_args.args[0]
    assert args.image_dir == tmp_path
    assert args.output_path == out


def test_create_existing_dataset_does_not_import_handler(tmp_path: Path):
    # bail out early if dataset file already exists
    out = tmp_path / "data.h5"
    out.touch()
    with (
        patch("iconoscope.cli.import_module") as importer,
        patch(
            "sys.argv",
            ["iconoscope", "create", str(tmp_path), str(out)],
        ),
        pytest.raises(SystemExit),
    ):
        cli.main()

    importer.assert_not_called()


def test_create_reports_image_total(tmp_path: Path, capsys):
    out = tmp_path / "data.h5"
    from iconoscope.commands import create as create_command

    with patch("iconoscope.commands.create.ImageDataset.create") as create:
        dataset = create.return_value
        dataset.info.return_value = {"image_paths": 12}
        create_command.main(
            argparse.Namespace(image_dir=tmp_path, output_path=out, max=None)
        )

    assert "12" in capsys.readouterr().out


def test_cli_lazy_load():
    # for speed, calling the cli should not import heavy dependencies
    code = """
import sys
import time
started = time.perf_counter()
import iconoscope.cli

heavy = {"torch", "transformers", "umap", "sklearn"}
loaded = heavy.intersection(sys.modules)
assert not loaded, f"heavy dependencies loaded: {sorted(loaded)}"
print(f"{time.perf_counter() - started:.4f}")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    elapsed = float(result.stdout.strip())
    logger.info("iconoscope.cli import: %ss", elapsed)
    assert elapsed < MAX_CLI_IMPORT_SECONDS, (
        f"iconoscope.cli import took {elapsed:.4f}s; "
        f"expected under {MAX_CLI_IMPORT_SECONDS:.1f}s"
    )


## custom size type for argparse to support specifying size as wxh


def test_size_tuple():
    # single dimension -> square
    assert cli.size_tuple("200") == argparse.Namespace(width=200, height=200)
    assert cli.size_tuple(200) == argparse.Namespace(width=200, height=200)
    # width x height
    assert cli.size_tuple("150x250") == argparse.Namespace(width=150, height=250)

    # errors
    # too many dimensions
    with pytest.raises(ValueError, match="Could not parse"):
        cli.size_tuple("100x200x300")
    # too few
    with pytest.raises(ValueError, match="invalid literal"):
        cli.size_tuple("")
    # non-numeric
    with pytest.raises(ValueError, match="invalid literal"):
        cli.size_tuple("a")
    with pytest.raises(ValueError, match="invalid literal"):
        cli.size_tuple("10xb")


@pytest.mark.parametrize(
    "value, expected",
    [("60", 0.6), ("100", 1.0), ("0.5", 0.005), ("33.333333", 0.3333), ("12.5", 0.125)],
)
def test_percentage(value, expected):
    assert cli.percentage(value) == expected


@pytest.mark.parametrize("value", ["0", "-5", "100.1", "150", "abc"])
def test_percentage_invalid(value):
    with pytest.raises(argparse.ArgumentTypeError):
        cli.percentage(value)


def test_percentage_warns_below_one(capsys):
    assert cli.percentage("0.6") == 0.006
    err = capsys.readouterr().err
    assert "Warning: 0.6 is interpreted as 0.6% of the image" in err
    assert "use 60 for 60%" in err


@pytest.mark.parametrize("value", ["1", "60", "100"])
def test_percentage_no_warning(value, capsys):
    cli.percentage(value)
    assert capsys.readouterr().err == ""

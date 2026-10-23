import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from iconoscope import mosaic as mosaic_layout
from iconoscope.commands import mosaic
from iconoscope.dataset import ImageDataset
from iconoscope.models import DEFAULT_MODEL, EmbeddingConfig


def mosaic_args(dataset: Path, **opts) -> argparse.Namespace:
    # mirror the args provided by the CLI parser
    defaults = {
        "model": DEFAULT_MODEL.name,
        "center_crop": None,
        "grayscale": False,
        "config": None,
    }
    return argparse.Namespace(
        dataset=dataset,
        output=None,
        size=argparse.Namespace(width=100, height=100),
        jpeg_quality=90,
        limit=None,
        **(defaults | opts),
    )


@pytest.fixture
def dataset_with_features(tmp_image_dir: Path) -> ImageDataset:
    dataset = ImageDataset.create(tmp_image_dir, tmp_image_dir / "data.h5")
    df = dataset.get_image_data().select("image_path")
    dataset.save_features(
        df.with_columns(features=np.zeros((df.height, 4))),
        EmbeddingConfig(DEFAULT_MODEL, center_crop=0.5),
    )
    return dataset


def test_mosaic_requires_matching_features(dataset_with_features):
    with pytest.raises(SystemExit, match="No features"):
        mosaic.main(mosaic_args(dataset_with_features.storage_path))


@patch("iconoscope.commands.mosaic.generate_mosaic")
def test_mosaic_uses_embedding_config(mock_generate, dataset_with_features):
    mosaic.main(mosaic_args(dataset_with_features.storage_path, center_crop=0.5))
    mock_generate.assert_called_once()
    assert mock_generate.call_args.args[1] == EmbeddingConfig(
        DEFAULT_MODEL, center_crop=0.5
    )


@patch("iconoscope.commands.mosaic.generate_mosaic")
def test_mosaic_by_config_label(mock_generate, dataset_with_features):
    config = EmbeddingConfig(DEFAULT_MODEL, center_crop=0.5)
    mosaic.main(
        mosaic_args(dataset_with_features.storage_path, model=None, config=config.label)
    )
    assert mock_generate.call_args.args[1] == config


@patch("iconoscope.commands.mosaic.generate_mosaic")
def test_mosaic_defaults_to_only_stored_config(mock_generate, dataset_with_features):
    # no model or config specified; dataset only has center crop features
    mosaic.main(mosaic_args(dataset_with_features.storage_path, model=None))
    assert mock_generate.call_args.args[1] == EmbeddingConfig(
        DEFAULT_MODEL, center_crop=0.5
    )


def make_canvas_error(image_count, width, height, aspect_ratio=1.0):
    try:
        mosaic_layout.check_canvas_size(image_count, width, height, aspect_ratio)
    except mosaic_layout.CanvasTooSmallError as err:
        return err
    raise AssertionError("expected canvas to be too small")


def test_canvas_too_small_message_square():
    err = make_canvas_error(50000, 2000, 2000, 1.5)
    message = mosaic.canvas_too_small_message(err)
    assert str(err) in message
    width, _ = err.min_canvas
    assert (
        f"To fix, use --limit {err.max_images} or lower, or --size {width} or larger"
        in message
    )


def test_canvas_too_small_message_rectangle():
    err = make_canvas_error(20000, 2000, 200)
    width, height = err.min_canvas
    assert f"--size {width}x{height} or larger" in mosaic.canvas_too_small_message(err)


def test_canvas_too_small_message_no_images_fit():
    # canvas too small for even one image: only suggest a larger size
    err = make_canvas_error(10, 19, 19)
    message = mosaic.canvas_too_small_message(err)
    assert "--limit" not in message
    assert "To fix, use --size" in message
    assert "or larger" in message


@patch("iconoscope.commands.mosaic.generate_mosaic")
def test_mosaic_canvas_too_small_exits(mock_generate, dataset_with_features):
    err = make_canvas_error(50000, 2000, 2000, 1.5)
    mock_generate.side_effect = err
    with pytest.raises(SystemExit) as excinfo:
        mosaic.main(mosaic_args(dataset_with_features.storage_path, center_crop=0.5))
    assert excinfo.value.code == mosaic.canvas_too_small_message(err)

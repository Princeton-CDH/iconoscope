import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from iconoscope.commands import mosaic
from iconoscope.dataset import ImageDataset
from iconoscope.models import DEFAULT_MODEL, EmbeddingConfig


def mosaic_args(dataset: Path, **opts) -> argparse.Namespace:
    # mirror the args provided by the CLI parser
    defaults = {"model": DEFAULT_MODEL.name, "center_crop": None, "grayscale": False}
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

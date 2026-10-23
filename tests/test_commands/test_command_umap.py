import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from iconoscope.commands import umap
from iconoscope.dataset import ImageDataset
from iconoscope.models import DEFAULT_MODEL, EmbeddingConfig

GRAY = EmbeddingConfig(DEFAULT_MODEL, grayscale=True)
CROP = EmbeddingConfig(DEFAULT_MODEL, center_crop=0.5)


def umap_args(dataset: Path, **opts) -> argparse.Namespace:
    # mirror the args provided by the CLI parser (no model specified)
    defaults = {
        "model": None,
        "center_crop": None,
        "grayscale": False,
        "config": None,
        "update": False,
    }
    return argparse.Namespace(dataset=dataset, **(defaults | opts))


@pytest.fixture
def dataset_with_features(tmp_image_dir: Path) -> ImageDataset:
    dataset = ImageDataset.create(tmp_image_dir, tmp_image_dir / "data.h5")
    df = dataset.get_image_data().select("image_path")
    for config in (GRAY, CROP):
        dataset.save_features(
            df.with_columns(features=np.zeros((df.height, 4))), config
        )
    return dataset


def computed_configs(mock_compute) -> list[EmbeddingConfig]:
    return [call.args[1] for call in mock_compute.call_args_list]


@patch("iconoscope.commands.umap.get_or_compute_umap")
def test_umap_all_configs(mock_compute, dataset_with_features, capsys):
    umap.main(umap_args(dataset_with_features.storage_path))
    assert set(computed_configs(mock_compute)) == {GRAY, CROP}
    assert "calculating UMAP" in capsys.readouterr().out


@patch("iconoscope.commands.umap.get_or_compute_umap")
def test_umap_model_options(mock_compute, dataset_with_features):
    umap.main(umap_args(dataset_with_features.storage_path, grayscale=True))
    assert computed_configs(mock_compute) == [GRAY]


def test_umap_model_options_no_features(dataset_with_features):
    with pytest.raises(SystemExit, match="No features"):
        umap.main(
            umap_args(dataset_with_features.storage_path, model=DEFAULT_MODEL.name)
        )


@patch("iconoscope.commands.umap.get_or_compute_umap")
def test_umap_multiple_config_labels(mock_compute, dataset_with_features):
    # duplicates are ignored; order is preserved
    labels = [CROP.label, GRAY.label, CROP.label]
    umap.main(umap_args(dataset_with_features.storage_path, config=labels))
    assert computed_configs(mock_compute) == [CROP, GRAY]


def test_umap_unknown_config_label(dataset_with_features):
    with pytest.raises(SystemExit, match="No features for `bogus/defaults`"):
        umap.main(
            umap_args(dataset_with_features.storage_path, config=["bogus/defaults"])
        )


def test_umap_no_features(tmp_image_dir):
    dataset = ImageDataset.create(tmp_image_dir, tmp_image_dir / "data.h5")
    with pytest.raises(SystemExit, match="No features in dataset"):
        umap.main(umap_args(dataset.storage_path))


@patch("iconoscope.commands.umap.get_or_compute_umap")
def test_umap_skips_existing(mock_compute, dataset_with_features, capsys):
    dataset_with_features.save_umap(np.zeros((3, 2)), GRAY)
    umap.main(umap_args(dataset_with_features.storage_path))
    assert computed_configs(mock_compute) == [CROP]
    assert f"{GRAY.label}: UMAP already exists" in capsys.readouterr().out

    # with update, existing umap is recalculated
    mock_compute.reset_mock()
    umap.main(umap_args(dataset_with_features.storage_path, update=True))
    assert set(computed_configs(mock_compute)) == {GRAY, CROP}
    assert all(call.kwargs["overwrite"] for call in mock_compute.call_args_list)

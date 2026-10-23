import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from iconoscope.commands import cluster
from iconoscope.dataset import ImageDataset
from iconoscope.models import DEFAULT_MODEL, EmbeddingConfig


def cluster_args(dataset: Path, n_clusters: int = 2, **opts) -> argparse.Namespace:
    # mirror the args provided by the CLI parser
    defaults = {"model": DEFAULT_MODEL.name, "center_crop": None, "grayscale": False}
    return argparse.Namespace(
        dataset=dataset, n_clusters=n_clusters, **(defaults | opts)
    )


@pytest.fixture
def dataset_with_features(tmp_image_dir: Path) -> ImageDataset:
    dataset = ImageDataset.create(tmp_image_dir, tmp_image_dir / "data.h5")
    df = dataset.get_image_data().select("image_path")
    dataset.save_features(
        df.with_columns(features=np.zeros((df.height, 4))),
        EmbeddingConfig(DEFAULT_MODEL, grayscale=True),
    )
    return dataset


def test_cluster_requires_matching_features(dataset_with_features):
    # features only exist for grayscale config
    with pytest.raises(SystemExit, match="No features"):
        cluster.main(cluster_args(dataset_with_features.storage_path))


@patch("iconoscope.commands.cluster.identify_clusters")
def test_cluster_uses_embedding_config(mock_identify, dataset_with_features, capsys):
    mock_identify.return_value = np.array([0, 1, 1])
    config = EmbeddingConfig(DEFAULT_MODEL, grayscale=True)
    cluster.main(cluster_args(dataset_with_features.storage_path, grayscale=True))

    # clustering + saving use the preprocessing-specific configuration
    assert mock_identify.call_args.args[1] == config
    df = dataset_with_features.load_data(config, features=True, clusters=True)
    assert df["cluster_kmeans_2"].to_list() == [0, 1, 1]

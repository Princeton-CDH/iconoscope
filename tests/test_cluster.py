from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from iconoscope.cluster import identify_clusters
from iconoscope.models import DEFAULT_MODEL


def _features_df(features: np.ndarray) -> pl.DataFrame:
    """Build a features dataframe matching the shape returned by
    ImageDataset.load_data(features=True) (an Array column)."""
    n_dims = features.shape[1]
    return pl.DataFrame(
        {"features": features.astype("float32")},
        schema={"features": pl.Array(pl.Float32, n_dims)},
    )


def _mock_dataset(features: np.ndarray) -> MagicMock:
    mock_ds = MagicMock()
    mock_ds.load_data.return_value = _features_df(features)
    return mock_ds


def test_identify_clusters_returns_label_per_item():
    rng = np.random.default_rng(0)
    features = rng.random((10, 8))
    mock_ds = _mock_dataset(features)

    labels = identify_clusters(mock_ds, DEFAULT_MODEL, n_clusters=3)

    assert isinstance(labels, np.ndarray)
    assert len(labels) == 10
    # labels should only use values 0..n_clusters-1
    assert set(labels.tolist()) <= {0, 1, 2}


def test_identify_clusters_requests_features_only():
    rng = np.random.default_rng(1)
    features = rng.random((6, 4))
    mock_ds = _mock_dataset(features)

    identify_clusters(mock_ds, DEFAULT_MODEL, n_clusters=2)

    mock_ds.load_data.assert_called_once_with(model=DEFAULT_MODEL, features=True)


def test_identify_clusters_separates_distinct_groups():
    # two well-separated blobs should be split into two distinct clusters
    rng = np.random.default_rng(2)
    blob1 = rng.normal(loc=0.0, scale=0.05, size=(5, 4))
    blob2 = rng.normal(loc=10.0, scale=0.05, size=(5, 4))
    features = np.vstack([blob1, blob2])
    mock_ds = _mock_dataset(features)

    labels = identify_clusters(mock_ds, DEFAULT_MODEL, n_clusters=2)

    # all items in the first blob share a label, all items in second blob share
    # a (different) label
    assert len(set(labels[:5].tolist())) == 1
    assert len(set(labels[5:].tolist())) == 1
    assert labels[0] != labels[5]


def test_identify_clusters_respects_n_clusters():
    rng = np.random.default_rng(3)
    # four well separated blobs
    centers = [0.0, 20.0, 40.0, 60.0]
    blobs = [rng.normal(loc=center, scale=0.05, size=(5, 4)) for center in centers]
    features = np.vstack(blobs)
    mock_ds = _mock_dataset(features)

    labels = identify_clusters(mock_ds, DEFAULT_MODEL, n_clusters=4)

    assert len(set(labels.tolist())) == 4


def test_identify_clusters_is_deterministic():
    # random_state is fixed, so repeated calls on the same data should agree
    rng = np.random.default_rng(4)
    features = rng.random((12, 6))

    labels_a = identify_clusters(_mock_dataset(features), DEFAULT_MODEL, n_clusters=3)
    labels_b = identify_clusters(_mock_dataset(features), DEFAULT_MODEL, n_clusters=3)

    assert (labels_a == labels_b).all()


def test_identify_clusters_single_cluster():
    rng = np.random.default_rng(5)
    features = rng.random((6, 4))
    mock_ds = _mock_dataset(features)

    labels = identify_clusters(mock_ds, DEFAULT_MODEL, n_clusters=1)

    assert set(labels.tolist()) == {0}


def test_identify_clusters_n_clusters_greater_than_items():
    rng = np.random.default_rng(6)
    features = rng.random((3, 4))
    mock_ds = _mock_dataset(features)

    with pytest.raises(ValueError):
        identify_clusters(mock_ds, DEFAULT_MODEL, n_clusters=5)

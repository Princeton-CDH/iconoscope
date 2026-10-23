from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from iconoscope.dataset import ImageDataset
from iconoscope.models import DEFAULT_MODEL, EmbeddingConfig
from iconoscope.umap import MIN_UMAP_IMAGES, get_or_compute_umap, reduce_features

CONFIG = EmbeddingConfig(DEFAULT_MODEL)


def test_reduce_features_too_few():
    with pytest.raises(ValueError, match="at least"):
        reduce_features(np.zeros((MIN_UMAP_IMAGES - 1, 8), dtype=np.float32))


def test_get_or_compute_umap(tmp_path: Path, tmp_image_dir: Path):
    dataset = ImageDataset.create(
        storage_path=tmp_path / "data.h5", image_dir=tmp_image_dir
    )
    n = dataset.image_count
    df = dataset.get_image_data().with_columns(
        features=pl.Series(np.zeros((n, 4), dtype=np.float32))
    )
    dataset.save_features(df.select("image_path", "features"), CONFIG)
    coords = np.random.default_rng(0).random((n, 2))
    with patch("iconoscope.umap.reduce_features", return_value=coords) as reduce:
        result = get_or_compute_umap(dataset, CONFIG)
        assert reduce.call_count == 1
        assert np.allclose(result, coords)
        assert dataset.has_umap(CONFIG)
        # second call reads from storage
        assert np.allclose(get_or_compute_umap(dataset, CONFIG), coords)
        assert reduce.call_count == 1
        # overwrite recalculates
        get_or_compute_umap(dataset, CONFIG, overwrite=True)
        assert reduce.call_count == 2

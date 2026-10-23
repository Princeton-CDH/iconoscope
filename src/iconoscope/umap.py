import logging

import numpy as np
import umap
from sklearn.decomposition import PCA

from iconoscope.dataset import ImageDataset
from iconoscope.models import EmbeddingConfig

logger = logging.getLogger(__name__)

#: UMAP fails on very small inputs; not intended for tiny datasets
MIN_UMAP_IMAGES = 16


def reduce_features(features: np.ndarray) -> np.ndarray:
    """Takes an array of embedding feature vectors and returns normalized coordinates.
    Uses PCA to reduce, UMAP to transform to two dimensions, then normalize from 0 to 1 for
    both axes. Returns an array of x,y coordinates for each feature vector in the input."""
    if features.shape[0] < MIN_UMAP_IMAGES:
        raise ValueError(
            f"UMAP requires at least {MIN_UMAP_IMAGES} images; got {features.shape[0]}"
        )
    # use PCA to reduce vectors from 768 to 50 (but handle small datasets < 50)
    n_components = min(50, features.shape[0], features.shape[1])
    reduced = PCA(n_components=n_components).fit_transform(features)
    # use umap to project the reduced vectors into two dimensions
    coords = umap.UMAP(n_components=2).fit_transform(reduced)

    # determine smalleest and largest coordinates, and then
    # scale all coordinates to normalize from 0 to 1.0
    min_coords, max_coords = coords.min(0), coords.max(0)
    span = np.where(max_coords - min_coords > 0, max_coords - min_coords, 1.0)
    return (coords - min_coords) / span


def get_or_compute_umap(
    img_dataset: ImageDataset, config: EmbeddingConfig, overwrite: bool = False
) -> np.ndarray:
    """Return umap coordinates for a configuration from the dataset; if not
    stored (or ``overwrite`` is True), calculate from stored features via
    :func:`reduce_features` and save to the dataset."""
    if not overwrite and img_dataset.has_umap(config):
        logger.debug("Loading saved UMAP for %s", config.label)
        df = img_dataset.load_data(config, paths=False, umap=True)
        return df["umap"].to_numpy()
    logger.info("Calculating UMAP for %s", config.label)
    features = img_dataset.load_data(config, paths=False, features=True)["features"]
    coords = reduce_features(features.to_numpy())
    img_dataset.save_umap(coords, config)
    return coords

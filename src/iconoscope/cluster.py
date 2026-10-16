import numpy as np
from sklearn.cluster import KMeans

from iconoscope.dataset import ImageDataset
from iconoscope.models import EmbeddingConfig


def identify_clusters(
    img_dataset: ImageDataset, config: EmbeddingConfig, n_clusters: int
) -> np.ndarray:
    kmeans = KMeans(
        n_clusters=n_clusters,
        max_iter=100,
        n_init="auto",
        random_state=0,
    )
    df = img_dataset.load_data(config, features=True)
    # mini-batch kmeans allows fitting in batches, but for our collections are small for now to not need that
    # fit the data and predict cluster membership
    cluster_labels = kmeans.fit_predict(df["features"])
    return cluster_labels

import argparse

from iconoscope.cluster import identify_clusters
from iconoscope.commands import get_embedding_config, require_features
from iconoscope.dataset import ImageDataset


def main(args: argparse.Namespace) -> None:
    embed_config = get_embedding_config(args)

    img_dataset = ImageDataset(storage_path=args.dataset)
    require_features(img_dataset, embed_config)
    labels = identify_clusters(img_dataset, embed_config, args.n_clusters)
    # TODO: printing the labels is not very useful... summarize clustering somehow?
    print(labels)
    img_dataset.save_clusters(labels, args.n_clusters, embed_config)

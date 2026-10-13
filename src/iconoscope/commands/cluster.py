import argparse

from iconoscope.cluster import identify_clusters
from iconoscope.commands import get_model, require_features
from iconoscope.dataset import ImageDataset


def main(args: argparse.Namespace) -> None:
    model = get_model(args.model)
    img_dataset = ImageDataset(storage_path=args.dataset)
    require_features(img_dataset, model)
    labels = identify_clusters(img_dataset, model, args.n_clusters)
    # TODO: printing the labels is not very useful... summarize clustering somehow?
    print(labels)
    img_dataset.save_clusters(labels, args.n_clusters, model)

import argparse

from iconoscope.cluster import identify_clusters
from iconoscope.dataset import ImageDataset
from iconoscope.models import AVAILABLE_MODELS


def main(args: argparse.Namespace) -> None:
    try:
        model = AVAILABLE_MODELS[args.model]
    except KeyError:
        raise SystemExit("Unknown model `%s`", args.model)
    img_dataset = ImageDataset(storage_path=args.dataset)
    labels = identify_clusters(img_dataset, model, args.n_clusters)
    print(labels)
    img_dataset.save_clusters(labels, args.n_clusters, model)

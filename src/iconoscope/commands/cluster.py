import argparse

from iconoscope.cluster import identify_clusters
from iconoscope.commands import get_model, get_preprocess_opts, require_features
from iconoscope.dataset import ImageDataset


def main(args: argparse.Namespace) -> None:
    model = get_model(args.model)
    preprocess_opts = get_preprocess_opts(args)
    print(preprocess_opts)

    img_dataset = ImageDataset(storage_path=args.dataset)
    require_features(img_dataset, model, **preprocess_opts)
    labels = identify_clusters(img_dataset, model, args.n_clusters)
    # TODO: printing the labels is not very useful... summarize clustering somehow?
    print(labels)
    img_dataset.save_clusters(labels, args.n_clusters, model, **preprocess_opts)

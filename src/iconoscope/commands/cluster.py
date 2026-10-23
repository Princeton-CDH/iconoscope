import argparse
from collections import Counter

from iconoscope.cluster import identify_clusters
from iconoscope.commands import load_dataset, resolve_config


def main(args: argparse.Namespace) -> None:
    img_dataset = load_dataset(args.dataset)
    embed_config = resolve_config(img_dataset, args)
    labels = identify_clusters(img_dataset, embed_config, args.n_clusters)
    # TODO: printing the labels is not very useful... summarize clustering somehow?
    # use Counter to summarize the results
    img_dataset.save_clusters(labels, args.n_clusters, embed_config)

    label_totals = Counter(labels)
    print(
        "Totals per cluster: ",
        "; ".join(f"{n}: {label_totals[n]:,}" for n in sorted(label_totals.keys())),
    )

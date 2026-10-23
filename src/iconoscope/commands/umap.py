import argparse

from iconoscope.commands import load_dataset, resolve_configs
from iconoscope.umap import get_or_compute_umap


def main(args: argparse.Namespace) -> None:
    img_dataset = load_dataset(args.dataset)
    for config in resolve_configs(img_dataset, args):
        if not args.update and img_dataset.has_umap(config):
            print(f"{config.label}: UMAP already exists (use --update to recalculate)")
            continue
        print(f"{config.label}: calculating UMAP")
        get_or_compute_umap(img_dataset, config, overwrite=args.update)

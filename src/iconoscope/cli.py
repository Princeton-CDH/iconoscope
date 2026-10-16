import argparse
import sys
from importlib import import_module
from pathlib import Path

from iconoscope.models import AVAILABLE_MODELS, DEFAULT_MODEL


def size_tuple(size_str: str | int) -> argparse.Namespace:
    try:
        parts = [int(side) for side in str(size_str).split("x")]
    except ValueError as err:
        raise ValueError(f"Could not parse `{size_str}` as a size: {err}") from err
    # first portion is the width
    width = parts.pop(0)
    # if a second value is present, use as height; otherwise width
    height = parts.pop() if parts else width
    # if there are parts leftover, raise value error
    if parts:
        raise ValueError(f"Could not parse `{size_str}` as a size")
    return argparse.Namespace(width=width, height=height)


def existing_directory(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"{path} is not a directory")
    return path


def new_file(value: str) -> Path:
    path = Path(value)
    if path.exists():
        raise argparse.ArgumentTypeError(f"{path} already exists")
    return path


def percentage(value: str) -> float:
    """Parse a percentage greater than 0 and up to 100; returns it as a
    fraction (0.0-1.0), rounded to avoid floating point noise."""
    try:
        percent = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Could not convert `{value}` to percentage")
    if not 0.0 < percent <= 100.0:
        raise argparse.ArgumentTypeError(
            f"{value} must be greater than 0 and no more than 100"
        )
    if percent < 1.0:
        # likely a fraction entered by mistake (e.g. 0.6 instead of 60)
        print(
            f"Warning: {value} is interpreted as {value}% of the image; "
            f"use {percent * 100:g} for {percent * 100:g}%",
            file=sys.stderr,
        )
    return round(percent / 100, 4)


HANDLERS = {
    "create": ("iconoscope.commands.create", "main"),
    "embed": ("iconoscope.commands.embed", "main"),
    "info": ("iconoscope.commands.info", "main"),
    "mosaic": ("iconoscope.commands.mosaic", "main"),
    "cluster": ("iconoscope.commands.cluster", "main"),
}


def dispatch(args: argparse.Namespace) -> None:
    module_name, handler_name = HANDLERS[args.command]
    handler = getattr(import_module(module_name), handler_name)
    handler(args)


def main():
    parser = argparse.ArgumentParser(prog="iconoscope")
    subparsers = parser.add_subparsers(dest="command")

    ## create : discover and validate images into a dataset
    parser_create = subparsers.add_parser("create")
    parser_create.add_argument(
        "image_dir",
        type=existing_directory,
        help="Directory containing images to inventory (can be nested)",
    )
    parser_create.add_argument(
        "output_path",
        type=new_file,
        help="File path for the image dataset (.hdf5)",
    )
    parser_create.add_argument(
        "-m",
        "--max",
        type=int,
        help="Limit to specified number of images (arbitrary subset, not sorted)",
    )

    # model arguments
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument(
        "--model",
        choices=list(AVAILABLE_MODELS),
        default=DEFAULT_MODEL.name,
        help="Embedding model (default: %(default)s)",
    )
    model_parser.add_argument(
        "--center-crop",
        type=percentage,
        metavar="CROP",
        help=(
            "Center crop images before embedding: percentage of image to retain, "
            "greater than 0 and up to 100 (e.g., 60 to use central 60%% of image). "
            "Crop is square, relative to the shortest edge, and replaces the "
            "model's default resizing and cropping."
        ),
    )
    model_parser.add_argument(
        "--grayscale",
        action="store_true",
        default=False,
        help="Convert images to grayscale before generate embeddings",
    )

    ## embed : extract features from an existing dataset
    parser_embed = subparsers.add_parser("embed", parents=[model_parser])
    parser_embed.add_argument(
        "dataset",
        type=Path,
        help="Image dataset file created by iconoscope create (.hdf5)",
    )
    parser_embed.add_argument(
        "-u",
        "--update",
        action="store_true",
        default=False,
        help="Overwrite features for the specified model even if they already exist "
        "(also removes any umap/cluster results for that model)",
    )

    ## dataset info
    info_parser = subparsers.add_parser("info")
    info_parser.add_argument(
        "dataset",
        type=Path,
        help="Image dataset file created by iconoscope embed (.hdf5)",
    )

    ## generate mosaic
    parser_mosaic = subparsers.add_parser("mosaic", parents=[model_parser])
    parser_mosaic.add_argument(
        "dataset",
        type=Path,
        help="Image dataset file produced by the embed command (.hdf5)",
    )
    parser_mosaic.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output image path (default: embeddings stem + .jpg)",
    )

    parser_mosaic.add_argument(
        "-s",
        "--size",
        type=size_tuple,
        default="2000",
        help="Output image size; specify a single dimension for a square or both width and height as 1000x250. (default: %(default)s)",
    )
    parser_mosaic.add_argument(
        "--jpeg-quality", type=int, default=90, dest="jpeg_quality"
    )
    parser_mosaic.add_argument(
        "-l",
        "--limit",
        type=int,
        help="Limit to sample of images of the specified size",
    )

    ## cluster image features
    cluster_parser = subparsers.add_parser("cluster", parents=[model_parser])
    cluster_parser.add_argument(
        "dataset",
        type=Path,
        help="Image dataset file created by iconoscope embed (.hdf5)",
    )
    cluster_parser.add_argument("n_clusters", type=int, help="Number of clusters")
    parser.set_defaults(func=dispatch)

    # Validate embed sources before lazy-loading the embedding stack.
    args = parser.parse_args()
    if not args.command:
        parser.error("a command is required")
    if args.command == "embed":
        if not args.dataset.is_file():
            parser.error(f"{args.dataset} is not an existing dataset file")
    args.func(args)


if __name__ == "__main__":
    main()

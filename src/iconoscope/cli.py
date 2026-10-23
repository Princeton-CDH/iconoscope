import argparse
import sys
from importlib import import_module
from pathlib import Path

from iconoscope.models import AVAILABLE_MODELS, DEFAULT_MODEL


#: minimum width/height in pixels for generated images
MIN_IMAGE_SIZE = 100


def size_tuple(size_str: str | int) -> argparse.Namespace:
    """Parse a size as a single dimension (square) or WIDTHxHEIGHT.
    Both dimensions must be at least :data:`MIN_IMAGE_SIZE` pixels."""
    try:
        parts = [int(side) for side in str(size_str).split("x")]
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            f"Could not parse `{size_str}` as a size: {err}"
        ) from err
    # first portion is the width
    width = parts.pop(0)
    # if a second value is present, use as height; otherwise width
    height = parts.pop() if parts else width
    # if there are parts leftover, too many dimensions were specified
    if parts:
        raise argparse.ArgumentTypeError(f"Could not parse `{size_str}` as a size")
    if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
        raise argparse.ArgumentTypeError(
            f"Size `{size_str}` is too small; width and height must be "
            f"at least {MIN_IMAGE_SIZE}"
        )
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


def int_range(
    value: str, min_value: int | None = None, max_value: int | None = None
) -> int:
    """Parse an integer and check that it is within the specified range
    (inclusive); either bound may be omitted."""
    try:
        integer = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Could not convert `{value}` to integer")
    if (min_value is not None and integer < min_value) or (
        max_value is not None and integer > max_value
    ):
        if max_value is None:
            expected = f"{min_value} or more"
        elif min_value is None:
            expected = f"{max_value} or less"
        else:
            expected = f"between {min_value} and {max_value}"
        raise argparse.ArgumentTypeError(f"{value} must be an integer {expected}")
    return integer


def positive_int(value: str) -> int:
    """Parse a positive integer (1 or more)."""
    return int_range(value, min_value=1)


#: default JPEG quality for generated images
DEFAULT_JPEG_QUALITY = 90
#: JPEG quality above this value increases file size with little visible benefit
MAX_RECOMMENDED_JPEG_QUALITY = 95
#: file extensions that are saved as JPEG (and use the quality setting)
JPEG_SUFFIXES = {".jpg", ".jpeg"}


def jpeg_quality(value: str) -> int:
    """Parse JPEG quality as an integer from 1 to 100; warns if the
    value is above the recommended maximum."""
    quality = int_range(value, min_value=1, max_value=100)
    if quality > MAX_RECOMMENDED_JPEG_QUALITY:
        print(
            f"Warning: JPEG quality above {MAX_RECOMMENDED_JPEG_QUALITY} "
            "greatly increases file size with little visible improvement",
            file=sys.stderr,
        )
    return quality


# to keep the CLI responsive, subcommands are separate and only imported when needed
HANDLERS = {
    "create": ("iconoscope.commands.create", "main"),
    "embed": ("iconoscope.commands.embed", "main"),
    "info": ("iconoscope.commands.info", "main"),
    "mosaic": ("iconoscope.commands.mosaic", "main"),
    "cluster": ("iconoscope.commands.cluster", "main"),
}


def dispatch(args: argparse.Namespace) -> None:
    # Import and run the specified subcommand with the specified arguments
    module_name, handler_name = HANDLERS[args.command]
    handler = getattr(import_module(module_name), handler_name)
    handler(args)


def main():
    parser = argparse.ArgumentParser(prog="iconoscope")
    subparsers = parser.add_subparsers(dest="command")

    # limit option shared by create and mosaic options
    limit_parser = argparse.ArgumentParser(add_help=False)
    limit_parser.add_argument(
        "-l",
        "--limit",
        type=positive_int,
        help="Limit to the specified number of images (arbitrary subset, not sorted)",
    )

    ## create : discover and validate images into a dataset
    parser_create = subparsers.add_parser("create", parents=[limit_parser])
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

    # model arguments
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument(
        "-m",
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
    parser_mosaic = subparsers.add_parser(
        "mosaic", parents=[model_parser, limit_parser]
    )
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
        "--jpeg-quality",
        type=jpeg_quality,
        # default is None to detect when quality is set for non-jpeg output;
        # set to DEFAULT_JPEG_QUALITY after parsing
        default=None,
        dest="jpeg_quality",
        help=f"JPEG quality, 1-100 (default: {DEFAULT_JPEG_QUALITY}); "
        "ignored for other output formats",
    )

    ## cluster image features
    cluster_parser = subparsers.add_parser("cluster", parents=[model_parser])
    cluster_parser.add_argument(
        "dataset",
        type=Path,
        help="Image dataset file created by iconoscope embed (.hdf5)",
    )
    cluster_parser.add_argument(
        "n_clusters", type=positive_int, help="Number of clusters"
    )
    parser.set_defaults(func=dispatch)

    # Validate embed sources before lazy-loading the embedding stack.
    args = parser.parse_args()
    if not args.command:
        parser.error("You must specify which command to run")
    if args.command == "embed":
        if not args.dataset.is_file():
            parser.error(f"{args.dataset} is not an existing dataset file")
    if args.command == "mosaic":
        if args.jpeg_quality is None:
            args.jpeg_quality = DEFAULT_JPEG_QUALITY
        elif (
            args.output is not None and args.output.suffix.lower() not in JPEG_SUFFIXES
        ):
            print(
                f"Warning: --jpeg-quality is ignored for non-JPEG output "
                f"({args.output.name})",
                file=sys.stderr,
            )
    args.func(args)


if __name__ == "__main__":
    main()

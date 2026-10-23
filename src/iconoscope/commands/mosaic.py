import argparse

from iconoscope.commands import get_embedding_config, require_features
from iconoscope.dataset import ImageDataset
from iconoscope.mosaic import CanvasTooSmallError, generate_mosaic


def canvas_too_small_message(err: CanvasTooSmallError) -> str:
    """Error message for the command line, with options that would fix it.

    Suggested values are known to fit, but nearby values may not: the grid
    is chosen to maximize cell size, so fit is not strictly monotonic in image
    count or canvas size."""
    width, height = err.min_canvas
    size = str(width) if width == height else f"{width}x{height}"
    fixes = [f"--size {size}"]
    if err.max_images > 0:
        fixes.insert(0, f"--limit {err.max_images}")
    return f"{err}\nTry {' or '.join(fixes)}"


def main(args: argparse.Namespace) -> None:
    embed_config = get_embedding_config(args)

    img_dataset = ImageDataset(storage_path=args.dataset)
    require_features(img_dataset, embed_config)
    try:
        generate_mosaic(
            img_dataset,
            embed_config,
            output=args.output,
            width=args.size.width,
            height=args.size.height,
            jpeg_quality=args.jpeg_quality,
            sample_size=args.limit,
        )
    except CanvasTooSmallError as err:
        raise SystemExit(canvas_too_small_message(err))

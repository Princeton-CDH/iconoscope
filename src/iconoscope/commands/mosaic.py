import argparse

from iconoscope.commands import load_dataset, resolve_config
from iconoscope.mosaic import CanvasTooSmallError, generate_mosaic


def canvas_too_small_message(err: CanvasTooSmallError) -> str:
    """Error message for the command line, with options that would fix it:
    any limit up to the suggested value, or any size at least as large as
    the suggested value (with the same proportions)."""
    width, height = err.min_canvas
    size = str(width) if width == height else f"{width}x{height}"
    fixes = [f"--size {size} or larger"]
    if err.max_images > 0:
        fixes.insert(0, f"--limit {err.max_images:d} or lower")
    return f"{err}\nTo fix, use {', or '.join(fixes)}"


def main(args: argparse.Namespace) -> None:
    img_dataset = load_dataset(args.dataset)
    embed_config = resolve_config(img_dataset, args)
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

import argparse

from iconoscope.dataset import ImageDataset
from iconoscope.models import AVAILABLE_MODELS
from iconoscope.mosaic import generate_mosaic


def main(args: argparse.Namespace) -> None:
    try:
        model = AVAILABLE_MODELS[args.model]
    except KeyError:
        raise SystemExit("Unknown model `%s`", args.model)

    img_dataset = ImageDataset(storage_path=args.dataset)
    generate_mosaic(
        img_dataset,
        model,
        output=args.output,
        width=args.size.width,
        height=args.size.height,
        jpeg_quality=args.jpeg_quality,
        sample_size=args.limit,
    )

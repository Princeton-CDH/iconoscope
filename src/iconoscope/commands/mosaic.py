import argparse

from iconoscope.commands import get_model, get_preprocess_opts, require_features
from iconoscope.dataset import ImageDataset
from iconoscope.mosaic import generate_mosaic


def main(args: argparse.Namespace) -> None:
    model = get_model(args.model)
    preprocess_opts = get_preprocess_opts(args)

    img_dataset = ImageDataset(storage_path=args.dataset)
    require_features(img_dataset, model, **preprocess_opts)
    generate_mosaic(
        img_dataset,
        model,
        output=args.output,
        width=args.size.width,
        height=args.size.height,
        jpeg_quality=args.jpeg_quality,
        sample_size=args.limit,
        **preprocess_opts,
    )

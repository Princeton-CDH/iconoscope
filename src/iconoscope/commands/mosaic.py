import argparse

from iconoscope.commands import get_embedding_config, require_features
from iconoscope.dataset import ImageDataset
from iconoscope.mosaic import generate_mosaic


def main(args: argparse.Namespace) -> None:
    embed_config = get_embedding_config(args)

    img_dataset = ImageDataset(storage_path=args.dataset)
    require_features(img_dataset, embed_config)
    generate_mosaic(
        img_dataset,
        embed_config,
        output=args.output,
        width=args.size.width,
        height=args.size.height,
        jpeg_quality=args.jpeg_quality,
        sample_size=args.limit,
    )

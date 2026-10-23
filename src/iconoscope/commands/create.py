import argparse

from iconoscope.dataset import ImageDataset


def main(args: argparse.Namespace) -> None:
    try:
        print(f"Discovering and validating images in {args.image_dir}...")
        dataset = ImageDataset.create(
            args.image_dir,
            args.output_path,
            max_images=args.limit,
        )
        total = dataset.info()["image_paths"]
        print(f"Created {args.output_path} with {total:,} images.")
    except ValueError as err:
        raise SystemExit(err)

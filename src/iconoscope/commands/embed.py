import argparse
import time

from iconoscope.commands import get_embedding_config
from iconoscope.dataset import ImageDataset
from iconoscope.embed import extract_img_features


def main(args: argparse.Namespace) -> None:
    embed_config = get_embedding_config(args)
    try:
        img_dataset = ImageDataset(storage_path=args.dataset)

        if img_dataset.has_features(embed_config) and not args.update:
            print(
                f"Dataset already has features for `{embed_config.model.name}/{embed_config.key}`; specify --update to regenerate"
            )
            return

        start_time = time.perf_counter()
        img_feature_df = extract_img_features(img_dataset, embed_config)
        duration = time.perf_counter() - start_time
        print(
            f"Successfully extracted features from {img_feature_df.height:,} images in {duration:.2f}s"
        )
        img_dataset.save_features(
            img_feature_df,
            embed_config,
            overwrite=args.update,
        )
    except ValueError as err:
        raise SystemExit(err)

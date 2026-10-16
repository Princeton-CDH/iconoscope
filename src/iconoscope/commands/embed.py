import argparse
import time

from iconoscope.commands import get_model, get_preprocess_opts
from iconoscope.dataset import ImageDataset
from iconoscope.embed import extract_img_features


def main(args: argparse.Namespace) -> None:
    model = get_model(args.model)
    preprocess_opts = get_preprocess_opts(args)
    print(preprocess_opts)
    try:
        img_dataset = ImageDataset(storage_path=args.dataset)

        if img_dataset.has_features(model, **preprocess_opts) and not args.update:
            print(
                f"Dataset already has features for `{model.name}`; specify --update to regenerate"
            )
            return

        start_time = time.perf_counter()
        img_feature_df = extract_img_features(img_dataset, model, **preprocess_opts)
        duration = time.perf_counter() - start_time
        print(
            f"Successfully extracted features from {img_feature_df.height:,} images in {duration:.2f}s"
        )
        img_dataset.save_features(
            img_feature_df, model, overwrite=args.update, **preprocess_opts
        )
    except ValueError as err:
        raise SystemExit(err)

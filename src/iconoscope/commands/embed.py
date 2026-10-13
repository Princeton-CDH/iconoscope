import argparse
import time

from iconoscope.dataset import ImageDataset
from iconoscope.embed import extract_img_features
from iconoscope.models import AVAILABLE_MODELS


def main(args: argparse.Namespace) -> None:
    try:
        model = AVAILABLE_MODELS[args.model]
    except KeyError:
        raise SystemExit("Unknown model `%s`", args.model)
    try:
        img_dataset = ImageDataset(storage_path=args.dataset)

        if img_dataset.has_features(model):
            print(f"Dataset already has features for `{model.name}`")
            return

        start_time = time.perf_counter()
        img_feature_df = extract_img_features(img_dataset, model)
        duration = time.perf_counter() - start_time
        print(
            f"Successfully extracted features from {img_feature_df.height:,} images in {duration:.2f}s"
        )
        img_dataset.save_features(img_feature_df, model)
    except ValueError as err:
        raise SystemExit(err)

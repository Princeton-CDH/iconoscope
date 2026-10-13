import logging

import polars as pl
import torch
import transformers
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from iconoscope.dataset import ImageDataset
from iconoscope.models import EmbeddingModel

logger = logging.getLogger(__name__)


class ExpectedUnexpectedKeysFilter(logging.Filter):
    """Suppress the transformers load report when the only problems it
    lists are UNEXPECTED keys matching known-safe prefixes (e.g. the
    classifier head of a classification checkpoint). Any other report
    content (MISSING, MISMATCH, other UNEXPECTED keys) is still logged."""

    def __init__(self, prefixes: tuple[str, ...]):
        super().__init__()
        self.prefixes = prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "LOAD REPORT" not in msg:
            return True
        status_rows = [
            line
            for line in msg.splitlines()
            if any(
                s in line for s in ("UNEXPECTED", "MISSING", "MISMATCH", "CONVERSION")
            )
            and "|" in line
        ]
        for row in status_rows:
            if "UNEXPECTED" not in row:
                return True
            key = row.split("|", 1)[0].strip()
            if not key.startswith(self.prefixes):
                return True
        # every flagged key is an expected unused key; drop the report
        logger.debug("Suppressed expected load report: %s", msg)
        return False


def extract_img_features(
    img_dataset: ImageDataset, model: EmbeddingModel
) -> pl.DataFrame:
    """ "
    Takes an image dataset and extracts features. Returns  a DataFrame with image_path, features.
    """
    # params to add later: model, boolean for progress bar

    # autodetect which device to use
    device = Accelerator().device
    logger.info("Using device %s", device)

    # get the model class; use AutoModel if not specified
    model_class = transformers.AutoModel
    if model.hf_model_class is not None:
        model_class = getattr(transformers, model.hf_model_class, None)
        if model_class is None:
            raise ValueError(
                f"Unknown transformers model class '{model.hf_model_class}' "
                f"for model '{model.name}'"
            )

    processor = transformers.AutoImageProcessor.from_pretrained(model.id)
    hf_logger = logging.getLogger("transformers.modeling_utils")
    load_filter = ExpectedUnexpectedKeysFilter(model.ignore_unexpected)
    if model.ignore_unexpected:
        hf_logger.addFilter(load_filter)
    try:
        embed_model = model_class.from_pretrained(model.id).to(device)
    finally:
        hf_logger.removeFilter(load_filter)

    batch_size = 256
    dataloader = DataLoader(
        img_dataset,
        batch_size=batch_size,
        collate_fn=ImageDataset.collate,
    )

    img_feature_df = pl.DataFrame(
        schema={
            "image_path": pl.String,
            "features": pl.Array(pl.Float32, model.dimensions),
        },
    )

    # batch size depends on model and available GPU/CPU memory
    # hf chat agent suggestion for ViT-Base with 224×224 images:
    # 8 GB -> 8–16; 16 GB -> 32–64; 24 GB -> 64–128; 40+ GB -> 128–256+

    progbar = tqdm(desc="Extracting features", total=img_dataset.image_count)
    for images, paths in dataloader:
        inputs = processor(images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = embed_model(**inputs)
            # some models (e.g. ResNet) return pooled output as (N, C, 1, 1);
            # flatten to (N, C) so it matches the expected embedding shape
            results = outputs.pooler_output.flatten(start_dim=1).cpu()
            img_feature_df.extend(
                pl.DataFrame(
                    data={
                        "image_path": paths,
                        "features": results,
                    }
                )
            )
            # update progress bar (how many to increase, not the total count)
            progbar.update(len(images))

    progbar.close()
    return img_feature_df

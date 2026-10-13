import logging

import polars as pl
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from iconoscope.dataset import ImageDataset
from iconoscope.models import EmbeddingModel

logger = logging.getLogger(__name__)


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

    processor = AutoImageProcessor.from_pretrained(model.id)
    embed_model = AutoModel.from_pretrained(model.id).to(device)

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
            results = outputs.pooler_output.cpu()
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

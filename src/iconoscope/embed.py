import logging

import polars as pl
import torch
import transformers
from accelerate import Accelerator
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import center_crop
from tqdm import tqdm

from iconoscope.dataset import ImageDataset
from iconoscope.models import EmbeddingConfig

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


def model_input_size(processor) -> tuple[int, int]:
    """Determine the (height, width) of the images the model's processor
    produces: the crop size if the processor center-crops, otherwise its
    resize target."""
    if getattr(processor, "do_center_crop", False) and processor.crop_size:
        size = processor.crop_size
    else:
        size = processor.size
    # size may be a dict or a SizeDict, depending on transformers version
    size = dict(size.items()) if hasattr(size, "items") else vars(size)
    if size.get("height") and size.get("width"):
        return (size["height"], size["width"])
    if size.get("shortest_edge"):
        return (size["shortest_edge"], size["shortest_edge"])
    raise ValueError(f"Could not determine model input size from {size}")


class CenterCropPercent:
    """Crop to the center portion of an image, relative to the model input
    size (same approach as timm/ConvNeXt `crop_pct`): resize so the image
    covers the target size divided by the fraction, then center-crop to the
    target size. The output is exactly the model input size, so the model
    sees the central `fraction` of the image (based on the shortest edge)."""

    def __init__(self, fraction: float, target_size: tuple[int, int]):
        self.fraction = fraction
        self.target_size = target_size

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        target_h, target_w = self.target_size
        # scale so the image covers the enlarged target in both dimensions
        scale = max(target_h / height, target_w / width) / self.fraction
        resized = img.resize(
            (max(target_w, round(width * scale)), max(target_h, round(height * scale))),
            Image.Resampling.BICUBIC,
        )
        return center_crop(resized, list(self.target_size))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.fraction * 100:g}% "
            f"of {self.target_size[1]}x{self.target_size[0]})"
        )


def extract_img_features(
    img_dataset: ImageDataset, config: EmbeddingConfig
) -> pl.DataFrame:
    """
    Takes an image dataset and extracts features using the model and preprocessing
    options in the specified configuration. Returns a DataFrame with image_path, features.

    If center crop is specified, customizes image preprocessing to retain the specified
    central portion of the image (fraction, 0.0-1.0, relative to the shortest edge) at the
    model input size, replacing any resizing and cropping done by the model's processor.
    """
    model = config.model
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

    transforms = None
    transform_ops = []
    # additional options to pass to the processor
    processor_opts = {}
    if config.center_crop is not None:
        # Many processors (e.g. DINOv2, CLIP, ResNet) also resize and center crop,
        # which would crop the custom crop further. Output the custom crop at the
        # model input size and disable the processor's resizing and cropping.
        transform_ops.append(
            CenterCropPercent(config.center_crop, model_input_size(processor))
        )
        processor_opts = {"do_resize": False, "do_center_crop": False}
    if config.grayscale:
        transform_ops.append(v2.Grayscale())

    if transform_ops:
        transforms = v2.Compose(transform_ops)

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
        # apply pre-processing transforms if any
        if transforms is not None:
            images = [transforms(img) for img in images]
        inputs = processor(images, return_tensors="pt", **processor_opts).to(device)
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

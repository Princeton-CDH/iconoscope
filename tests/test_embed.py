from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from iconoscope import embed
from iconoscope.dataset import ImageDataset
from iconoscope.models import DEFAULT_MODEL, EmbeddingConfig, EmbeddingModel

CONFIG = EmbeddingConfig(DEFAULT_MODEL)


def _fake_processor_and_model():
    """Build fake processor + model that produce correctly-shaped tensors.

    The fake processor records the batch size so the fake model can return a
    ``pooler_output`` of shape ``(batch_n, 768)`` for each batch.
    """

    def fake_processor(images, return_tensors="pt"):
        # inputs must support `**inputs` unpacking -> return a dict
        inputs = {"n": len(images)}
        # `.to(device)` returns the dict unchanged
        return MagicMock(to=MagicMock(return_value=inputs))

    class FakeModel:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.eval_mode = True
            return self

        def __call__(self, n):
            return MagicMock(
                pooler_output=torch.zeros((n, 768)),
            )

    return fake_processor, FakeModel()


def _patch_ml(processor, model):
    """Context managers patching the heavy ML deps on iconoscope.embed."""
    accelerator = MagicMock()
    accelerator.return_value.device = "cpu"
    return (
        patch.object(embed, "Accelerator", accelerator),
        patch.object(
            embed.transformers.AutoImageProcessor,
            "from_pretrained",
            return_value=processor,
        ),
        patch.object(
            embed.transformers.AutoModel, "from_pretrained", return_value=model
        ),
    )


def _persisted_dataset(tmp_image_dir: Path) -> ImageDataset:
    return ImageDataset.create(tmp_image_dir, tmp_image_dir / "a.h5")


def test_extract_returns_features(tmp_image_dir: Path):
    processor, model = _fake_processor_and_model()

    patches = _patch_ml(processor, model)
    with (
        patches[0],
        patches[1],
        patches[2],
        patch.object(embed, "tqdm") as progress,
    ):
        df = embed.extract_img_features(_persisted_dataset(tmp_image_dir), CONFIG)

    progress.assert_called_once_with(desc="Extracting features", total=3)
    assert df.height == 3
    assert df.columns == ["image_path", "features"]
    assert df["features"].dtype.size == DEFAULT_MODEL.dimensions


def test_extract_uses_accelerator_device(tmp_image_dir: Path):
    processor, model = _fake_processor_and_model()

    accelerator = MagicMock()
    accelerator.return_value.device = "meta-device"
    with (
        patch.object(embed, "Accelerator", accelerator),
        patch.object(
            embed.transformers.AutoImageProcessor,
            "from_pretrained",
            return_value=processor,
        ),
        patch.object(
            embed.transformers.AutoModel, "from_pretrained", return_value=model
        ),
    ):
        embed.extract_img_features(_persisted_dataset(tmp_image_dir), CONFIG)

    # model was moved onto the accelerator device
    assert model.device == "meta-device"


def test_extract_image_paths_match(tmp_image_dir: Path):
    processor, model = _fake_processor_and_model()

    img_dataset = _persisted_dataset(tmp_image_dir)

    patches = _patch_ml(processor, model)
    with patches[0], patches[1], patches[2]:
        df = embed.extract_img_features(img_dataset, CONFIG)

    expected = {path for _img, path in img_dataset}
    assert set(df["image_path"].to_list()) == expected


def test_extract_uses_hf_model_class(tmp_image_dir: Path):
    processor, model = _fake_processor_and_model()
    custom_model = EmbeddingModel(
        "clip", "openai/clip-vit-base-patch32", 768, hf_model_class="CLIPVisionModel"
    )
    patches = _patch_ml(processor, model)
    with (
        patches[0],
        patches[1],
        patches[2] as auto_load,
        patch.object(
            embed.transformers.CLIPVisionModel, "from_pretrained", return_value=model
        ) as clip_load,
    ):
        embed.extract_img_features(
            _persisted_dataset(tmp_image_dir), EmbeddingConfig(custom_model)
        )

    clip_load.assert_called_once_with(custom_model.id)
    auto_load.assert_not_called()


def test_extract_unknown_hf_model_class(tmp_image_dir: Path):
    bad_model = EmbeddingModel("bad", "foo/bar", 768, hf_model_class="NoSuchModel")
    with pytest.raises(ValueError, match="Unknown transformers model class"):
        embed.extract_img_features(
            _persisted_dataset(tmp_image_dir), EmbeddingConfig(bad_model)
        )


class RecordingProcessor:
    """Fake image processor that records images and options it is called with.
    Mimics a processor that resizes and center crops to 20x20 (like DINOv2)."""

    def __init__(self, processor):
        self.processor = processor
        self.do_center_crop = True
        self.crop_size = SimpleNamespace(height=20, width=20)
        self.size = SimpleNamespace(shortest_edge=24)
        self.images = []
        self.opts = []

    def __call__(self, images, return_tensors="pt", **opts):
        self.images.extend(images)
        self.opts.append(opts)
        return self.processor(images, return_tensors=return_tensors)


def _extract_with_recording_processor(tmp_image_dir, **opts) -> RecordingProcessor:
    processor, model = _fake_processor_and_model()
    recorder = RecordingProcessor(processor)
    patches = _patch_ml(recorder, model)
    with patches[0], patches[1], patches[2]:
        embed.extract_img_features(
            _persisted_dataset(tmp_image_dir), EmbeddingConfig(DEFAULT_MODEL, **opts)
        )
    assert len(recorder.images) == 3
    return recorder


def test_extract_default_preprocessing(tmp_image_dir: Path):
    recorder = _extract_with_recording_processor(tmp_image_dir)
    # fixture images (32x24 RGB) passed unchanged; processor resizes + crops
    assert all(img.mode == "RGB" and img.size == (32, 24) for img in recorder.images)
    assert recorder.opts == [{}]


def test_extract_grayscale(tmp_image_dir: Path):
    recorder = _extract_with_recording_processor(tmp_image_dir, grayscale=True)
    assert all(img.mode == "L" and img.size == (32, 24) for img in recorder.images)
    # model resizing and cropping still apply
    assert recorder.opts == [{}]


@pytest.mark.parametrize("center_crop", [0.5, 1.0])
def test_extract_center_crop_replaces_model_crop(tmp_image_dir: Path, center_crop):
    recorder = _extract_with_recording_processor(tmp_image_dir, center_crop=center_crop)
    # custom crop is resized to the model input size (crop size, 20x20) ...
    assert all(img.size == (20, 20) for img in recorder.images)
    # ... and the processor's own resizing and cropping is disabled
    assert recorder.opts == [{"do_resize": False, "do_center_crop": False}]


def _framed_image(width: int, height: int, border: float) -> Image.Image:
    """Blue image with a red frame `border` fraction wide on each side."""
    img = Image.new("RGB", (width, height), (255, 0, 0))
    bw, bh = round(width * border), round(height * border)
    img.paste((0, 0, 255), (bw, bh, width - bw, height - bh))
    return img


@pytest.mark.parametrize("img_size", [(400, 300), (300, 400), (300, 300)])
@pytest.mark.parametrize("target_size", [(224, 224), (20, 30)])
@pytest.mark.parametrize("fraction", [0.5, 0.8, 1.0])
def test_center_crop_percent_output_size(img_size, target_size, fraction):
    crop = embed.CenterCropPercent(fraction, target_size)
    # PIL size is (width, height); target size is (height, width)
    assert crop(Image.new("RGB", img_size)).size == (target_size[1], target_size[0])


def test_center_crop_percent_small_image():
    # images smaller than the target are upscaled, not padded
    crop = embed.CenterCropPercent(0.5, (224, 224))
    assert crop(Image.new("RGB", (10, 8))).size == (224, 224)


@pytest.mark.parametrize(
    "fraction, expect_border",
    [
        # frame is 10% of each side: 90% crop includes it, 70% crop does not
        (0.9, True),
        (0.7, False),
        # whole image: frame visible
        (1.0, True),
    ],
)
def test_center_crop_percent_retains_center(fraction, expect_border):
    img = _framed_image(300, 300, border=0.1)
    cropped = embed.CenterCropPercent(fraction, (100, 100))(img)
    corner = cropped.getpixel((1, 1))
    assert (corner[0] > 200) is expect_border
    # center is always retained
    assert cropped.getpixel((50, 50))[2] > 200


def test_center_crop_percent_relative_to_shortest_edge():
    # landscape image: fraction applies to the shortest edge (height); the
    # longer edge is additionally cropped to the square target
    img = _framed_image(400, 200, border=0.1)
    cropped = embed.CenterCropPercent(0.9, (100, 100))(img)
    # top edge (within 10% of height) is red; left edge is cropped away (blue)
    assert cropped.getpixel((50, 1))[0] > 200
    assert cropped.getpixel((1, 50))[2] > 200


def test_center_crop_percent_repr():
    assert repr(embed.CenterCropPercent(0.6, (224, 224))) == (
        "CenterCropPercent(60% of 224x224)"
    )


@pytest.mark.parametrize(
    "processor, expected",
    [
        # resize + center crop (DINOv2, CLIP): output is the crop size
        (
            SimpleNamespace(
                do_center_crop=True,
                crop_size=SimpleNamespace(height=224, width=224),
                size=SimpleNamespace(shortest_edge=256),
            ),
            (224, 224),
        ),
        # resize directly (SigLIP, DINOv3): output is the size
        (
            SimpleNamespace(
                do_center_crop=None, size=SimpleNamespace(height=384, width=256)
            ),
            (384, 256),
        ),
        # no do_center_crop attribute; shortest edge with internal crop (ResNet)
        (
            SimpleNamespace(
                size=SimpleNamespace(height=None, width=None, shortest_edge=224)
            ),
            (224, 224),
        ),
    ],
)
def test_model_input_size(processor, expected):
    assert embed.model_input_size(processor) == expected


def test_model_input_size_dict():
    # older transformers versions use plain dicts for sizes
    processor = SimpleNamespace(
        do_center_crop=True,
        crop_size={"height": 224, "width": 224},
        size={"shortest_edge": 256},
    )
    assert embed.model_input_size(processor) == (224, 224)


def test_model_input_size_crop_enabled_without_crop_size():
    processor = SimpleNamespace(
        do_center_crop=True, crop_size=None, size={"height": 384, "width": 384}
    )
    assert embed.model_input_size(processor) == (384, 384)


def test_model_input_size_unknown():
    with pytest.raises(ValueError, match="Could not determine model input size"):
        embed.model_input_size(SimpleNamespace(size=SimpleNamespace()))

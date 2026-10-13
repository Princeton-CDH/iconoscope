from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from iconoscope import embed
from iconoscope.dataset import ImageDataset
from iconoscope.models import DEFAULT_MODEL


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
            embed.AutoImageProcessor, "from_pretrained", return_value=processor
        ),
        patch.object(embed.AutoModel, "from_pretrained", return_value=model),
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
        df = embed.extract_img_features(
            _persisted_dataset(tmp_image_dir), DEFAULT_MODEL
        )

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
            embed.AutoImageProcessor, "from_pretrained", return_value=processor
        ),
        patch.object(embed.AutoModel, "from_pretrained", return_value=model),
    ):
        embed.extract_img_features(_persisted_dataset(tmp_image_dir), DEFAULT_MODEL)

    # model was moved onto the accelerator device
    assert model.device == "meta-device"


def test_extract_image_paths_match(tmp_image_dir: Path):
    processor, model = _fake_processor_and_model()

    img_dataset = _persisted_dataset(tmp_image_dir)

    patches = _patch_ml(processor, model)
    with patches[0], patches[1], patches[2]:
        df = embed.extract_img_features(img_dataset, DEFAULT_MODEL)

    expected = {path for _img, path in img_dataset}
    assert set(df["image_path"].to_list()) == expected

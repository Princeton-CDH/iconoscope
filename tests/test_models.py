from dataclasses import FrozenInstanceError

import pytest

from iconoscope.models import DEFAULT_MODEL, EmbeddingConfig


def test_option_names():
    # model is not a pre-processing option
    assert EmbeddingConfig.option_names() == ["center_crop", "grayscale"]


def test_opts():
    config = EmbeddingConfig(DEFAULT_MODEL, center_crop=0.6)
    assert config.opts == {"center_crop": 0.6, "grayscale": False}


@pytest.mark.parametrize(
    "opts, expected",
    [
        ({}, "defaults"),
        ({"grayscale": False}, "defaults"),
        ({"center_crop": None}, "defaults"),
        ({"grayscale": True}, "grayscale"),
        ({"center_crop": 0.6}, "center_crop=0.6"),
        ({"center_crop": 0.6, "grayscale": True}, "center_crop=0.6_grayscale"),
    ],
)
def test_key(opts, expected):
    assert EmbeddingConfig(DEFAULT_MODEL, **opts).key == expected


def test_option_labels_and_label():
    config = EmbeddingConfig(DEFAULT_MODEL, center_crop=0.6, grayscale=True)
    assert config.option_labels == ["center_crop=0.6", "grayscale"]
    assert config.label == f"{DEFAULT_MODEL.name}/center_crop=0.6_grayscale"
    assert EmbeddingConfig(DEFAULT_MODEL).option_labels == []


def test_key_excludes_model_name():
    # model name is stored separately as the parent group
    assert DEFAULT_MODEL.name not in EmbeddingConfig(DEFAULT_MODEL).key


def test_equal_configs_are_interchangeable():
    # frozen dataclass: equal + hashable, so usable as a dict/set key
    a = EmbeddingConfig(DEFAULT_MODEL, grayscale=True)
    b = EmbeddingConfig(DEFAULT_MODEL, grayscale=True)
    assert a == b
    assert len({a, b}) == 1
    assert a != EmbeddingConfig(DEFAULT_MODEL)


def test_frozen():
    config = EmbeddingConfig(DEFAULT_MODEL)
    with pytest.raises(FrozenInstanceError):
        config.grayscale = True  # type: ignore[misc]


@pytest.mark.parametrize(
    "center_crop, expected",
    [
        (0.6, "center_crop=0.6"),
        (0.25, "center_crop=0.25"),
        # float noise is formatted compactly
        (0.1 + 0.2, "center_crop=0.3"),
        (0.3333, "center_crop=0.3333"),
    ],
)
def test_key_float_format(center_crop, expected):
    assert EmbeddingConfig(DEFAULT_MODEL, center_crop=center_crop).key == expected


@pytest.mark.parametrize("center_crop", [0.0, -0.5, 1.5, 60.0])
def test_center_crop_invalid(center_crop):
    with pytest.raises(ValueError, match="center_crop must be a fraction"):
        EmbeddingConfig(DEFAULT_MODEL, center_crop=center_crop)


def test_center_crop_full_image_differs_from_default():
    # center_crop=1.0 keeps the whole image and disables the model's own
    # resize+crop, so it is distinct from the default (model) preprocessing
    config = EmbeddingConfig(DEFAULT_MODEL, center_crop=1.0)
    assert config.center_crop == 1.0
    assert config.key == "center_crop=1"
    assert config != EmbeddingConfig(DEFAULT_MODEL)

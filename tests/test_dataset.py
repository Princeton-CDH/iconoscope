import logging
from pathlib import Path
from typing import Iterable
from unittest.mock import patch

import h5py
import numpy as np
import polars as pl
import pytest
from PIL import Image

from iconoscope.dataset import ImageDataset, find_images
from iconoscope.store import StorageSchema
from iconoscope.models import DEFAULT_MODEL, EmbeddingConfig, EmbeddingModel

CONFIG = EmbeddingConfig(DEFAULT_MODEL)
CLIP_CONFIG = EmbeddingConfig(EmbeddingModel("clip", "foobar/clip", 73))

## image dataset class


def test_init_requires_existing_storage(tmp_path):
    with pytest.raises(ValueError, match=r"does not exist.*create\(\)"):
        ImageDataset(storage_path=tmp_path / "a.h5")

    # storage path exists (currently not validated)
    h5_datafile = tmp_path / "z.h5"
    h5_datafile.touch()
    assert ImageDataset(storage_path=h5_datafile)


def test_create_validation(tmp_path):
    image_dir = tmp_path / "images"
    # non-existent
    with pytest.raises(ValueError, match="not a directory"):
        ImageDataset.create(image_dir, tmp_path / "a.h5")
    # file instead of dir
    img_file = tmp_path / "img.png"
    img_file.touch()
    with pytest.raises(ValueError, match="not a directory"):
        ImageDataset.create(img_file, tmp_path / "a.h5")
    # no images
    image_dir.mkdir()
    with pytest.raises(ValueError, match="No valid images"):
        ImageDataset.create(image_dir, tmp_path / "a.h5")
    # storage already exists
    with pytest.raises(ValueError, match="already exists"):
        ImageDataset.create(image_dir, img_file)


def test_create_writes_validated_image_inventory(tmp_path: Path, tmp_image_dir: Path):
    # add a file with an image extension that will not pass verification
    (tmp_image_dir / "broken.png").write_text("not an image")
    dataset = ImageDataset.create(tmp_image_dir, tmp_path / "data.h5")

    assert list(dataset.get_image_paths()) == sorted(list(tmp_image_dir.glob("*.jpg")))
    assert dataset.image_count == 3
    with h5py.File(tmp_path / "data.h5", "r") as h5_file:
        assert h5_file["image/paths"].size == 3
        assert h5_file["image/sizes"].shape == (3, 2)
        assert "image/models" not in h5_file


def test_create_stores_absolute_image_dir_and_relative_paths(
    tmp_path: Path, tmp_image_dir: Path, monkeypatch
):
    # create from a relative image dir, as when running the CLI
    monkeypatch.chdir(tmp_image_dir.parent)
    dataset = ImageDataset.create(Path(tmp_image_dir.name), tmp_path / "data.h5")

    with h5py.File(tmp_path / "data.h5", "r") as h5_file:
        img_grp = h5_file["image"]
        assert img_grp.attrs["image_dir"] == str(tmp_image_dir.resolve())
        stored = img_grp["paths"][:].astype("T").tolist()
        assert stored == sorted(p.name for p in tmp_image_dir.glob("*.jpg"))

    # loaded paths are absolute and usable from any working directory
    monkeypatch.chdir(tmp_path.parent)
    paths = dataset.get_image_data()["image_path"].to_list()
    assert paths == sorted(str(p.resolve()) for p in tmp_image_dir.glob("*.jpg"))
    assert all(Path(p).is_file() for p in paths)


def test_get_image_data(tmp_path: Path, tmp_image_dir: Path):
    # when storage file doesn't exist, yields results from find image method
    h5_datafile = tmp_path / "data.h5"
    img_ds = ImageDataset.create(image_dir=tmp_image_dir, storage_path=h5_datafile)
    img_df = img_ds.get_image_data()
    assert img_df.columns == ["image_path", "image_size", "aspect_ratio"]
    assert img_df["image_path"].to_list() == sorted(
        list([str(p) for p in tmp_image_dir.glob("*jpg")])
    )
    # fixture images are 32x24
    assert img_df["image_size"].to_list() == [[32, 24]] * 3
    assert img_df["aspect_ratio"].to_list() == [32 / 24] * 3


def test_get_image_paths(tmp_path: Path, tmp_image_dir: Path):
    img_ds = ImageDataset.create(tmp_image_dir, tmp_path / "data.h5")
    img_paths = img_ds.get_image_paths()
    assert isinstance(img_paths, Iterable)
    img_paths = list(img_paths)
    # paths come from storage, sorted, as absolute Path objects
    assert img_paths == sorted(tmp_image_dir.glob("*.jpg"))
    assert all(isinstance(p, Path) for p in img_paths)

    test_image_paths = ["foobar_a.jpg", "b.jpg"]
    with patch.object(
        img_ds.store,
        "image_paths",
        return_value=[Path(p) for p in test_image_paths],
    ):
        assert [str(p) for p in img_ds.get_image_paths()] == test_image_paths


def test_iter(tmp_path: Path, tmp_image_dir: Path):
    h5_datafile = tmp_path / "data.h5"
    img_ds = ImageDataset.create(image_dir=tmp_image_dir, storage_path=h5_datafile)
    images = img_ds.__iter__()
    assert isinstance(images, Iterable)
    images = list(images)
    assert len(images) == 3  # matches fixture data
    # yield a tuple of image object and file path as string
    # yield img, str(file_path)
    for img_tuple in images:
        assert isinstance(img_tuple[0], Image.Image)
        assert isinstance(img_tuple[1], str)
        assert img_tuple[1].endswith(".jpg")


def test_iter_err(tmp_path: Path, tmp_image_dir: Path, caplog):
    caplog.set_level(logging.WARN)
    # handle image file that can't be loaded
    h5_datafile = tmp_path / "data.h5"
    # add a non-image file with an image extension
    bad_img = tmp_image_dir / "bogus.png"
    bad_img.write_text("this is not an image")
    img_ds = ImageDataset.create(image_dir=tmp_image_dir, storage_path=h5_datafile)
    images = list(img_ds.__iter__())
    assert len(images) == 3  # should return all fixtures but nothing else
    assert len(caplog.record_tuples) == 1
    assert "Skipping invalid image" in caplog.text  # warns about the problem


def test_collate_returns_lists():
    imgs = [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]
    paths = ["/a/1.jpg", "/a/2.jpg"]
    # batch is a list of tuples of image and image path
    batch = [(imgs[i], paths[i]) for i in range(len(imgs))]
    # collate returns list of images and list of image paths
    result_imgs, result_paths = ImageDataset.collate(batch)
    assert result_imgs == imgs
    assert result_paths == paths


def test_collate_single_item():
    img = Image.new("RGB", (4, 4))
    imgs, paths = ImageDataset.collate([(img, "/x.png")])
    assert len(imgs) == 1
    assert paths == ["/x.png"]


def test_save_features_preserves_existing_data(tmp_path: Path, tmp_image_dir: Path):
    storage_path = tmp_path / "data.h5"
    dataset = ImageDataset.create(
        storage_path=storage_path, image_dir=tmp_image_dir, max_images=2
    )
    # use the paths the dataset found; filesystem glob order varies by platform
    paths = [str(p) for p in dataset.get_image_paths()]

    dataset.save_features(
        pl.DataFrame(
            {
                "image_path": paths,
                "features": np.array([[1.0, 0.0], [0.0, 1.0]]),
            }
        ),
        CONFIG,
    )
    with h5py.File(storage_path, "r+") as h5_file:
        model_group = h5_file[dataset.schema.model_group(CONFIG, absolute=True)]
        model_group.create_dataset(
            dataset.schema.umap_ds, data=[[0.1, 0.2], [0.3, 0.4]]
        )
    dataset.save_clusters(np.array([0, 1]), 2, CONFIG)

    dataset.save_features(
        pl.DataFrame(
            {
                "image_path": paths,
                "features": np.array([[2.0, 0.0], [0.0, 2.0]]),
            }
        ),
        CLIP_CONFIG,
    )

    with h5py.File(storage_path, "r") as h5_file:
        assert set(h5_file["image/models"]) == {"dinov2", "clip"}
        dinov2_model_grp = h5_file[dataset.schema.model_group(CONFIG, absolute=True)]
        assert dinov2_model_grp[dataset.schema.features_ds][:].tolist() == [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        assert dinov2_model_grp[dataset.schema.umap_ds][:].tolist() == [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
        cluster_ds = dinov2_model_grp[dataset.schema.clusters_group]["kmeans_2"]
        assert cluster_ds[:].tolist() == [0, 1]
        assert cluster_ds.attrs["n_clusters"] == 2


def test_save_features_rejects_different_image_paths(
    tmp_path: Path, tmp_image_dir: Path
):
    h5_file = tmp_path / "data.h5"
    dataset = ImageDataset.create(image_dir=tmp_image_dir, storage_path=h5_file)
    with pytest.raises(ValueError, match="image paths"):
        dataset.save_features(
            pl.DataFrame(
                {
                    # image paths don't match what is in tmp_image_dir
                    "image_path": ["/images/one.jpg", "/images/other.jpg"],
                    "features": np.array([[2.0, 0.0], [0.0, 2.0]]),
                }
            ),
            CONFIG,
        )


def test_save_features_rejects_mismatched_feature_rows(
    tmp_path: Path, tmp_image_dir: Path
):
    h5_file = tmp_path / "data.h5"
    dataset = ImageDataset.create(image_dir=tmp_image_dir, storage_path=h5_file)
    image_paths = sorted(list([str(p) for p in tmp_image_dir.glob("*.jpg")]))
    dataset.save_features(
        pl.DataFrame(
            {
                "image_path": image_paths,
                "features": np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.5]]),
            }
        ),
        CONFIG,
    )
    with pytest.raises(
        ValueError, match="cannot prune image paths when existing models are present"
    ):
        dataset.save_features(
            pl.DataFrame(
                {
                    # subset of valid paths but image paths can't be pruned once features are saved
                    "image_path": image_paths[:1],
                    "features": np.array([[2.0, 0.0]]),
                }
            ),
            CLIP_CONFIG,
        )


def test_save_features_rm_invalid_paths_on_first_model(
    tmp_path: Path, tmp_image_dir: Path
):
    h5_file = tmp_path / "data.h5"
    dataset = ImageDataset.create(image_dir=tmp_image_dir, storage_path=h5_file)
    image_paths = sorted(list([str(p) for p in tmp_image_dir.glob("*.jpg")]))

    dataset.save_features(
        pl.DataFrame(
            {
                "image_path": [image_paths[0]],
                "features": np.array([[2.0, 0.0]]),
            }
        ),
        CONFIG,
    )

    assert dataset.image_count == 1
    assert dataset.get_image_data()["image_path"].to_list() == [image_paths[0]]


def test_save_features_rejects_non_2d_features(tmp_path: Path, tmp_image_dir: Path):
    dataset = ImageDataset.create(
        image_dir=tmp_image_dir, storage_path=tmp_image_dir / "data.h5"
    )
    with pytest.raises(ValueError, match="2-dimensional"):
        dataset.save_features(
            pl.DataFrame(
                {
                    "image_path": ["/images/one.jpg", "/images/two.jpg"],
                    "features": [2.0, 3.0],
                }
            ),
            CONFIG,
        )


def test_save_features_rejects_unsupported_columns(tmp_path: Path, tmp_image_dir: Path):
    dataset = ImageDataset.create(
        image_dir=tmp_image_dir, storage_path=tmp_image_dir / "data.h5", max_images=2
    )
    # use the paths the dataset found; filesystem glob order varies by platform
    image_paths = [str(p) for p in dataset.get_image_paths()]
    with pytest.raises(ValueError, match="unsupported"):
        dataset.save_features(
            pl.DataFrame(
                {
                    "image_path": image_paths,
                    "features": np.array([[2.0, 0.0], [0.0, 2.0]]),
                    "label": ["a", "b"],
                }
            ),
            CONFIG,
        )


def test_save_features_no_overwrite(tmp_path: Path, tmp_image_dir: Path):
    dataset = ImageDataset.create(
        image_dir=tmp_image_dir, storage_path=tmp_image_dir / "data.h5", max_images=2
    )
    # use the paths the dataset found; filesystem glob order varies by platform
    image_paths = [str(p) for p in dataset.get_image_paths()]
    feature_df = pl.DataFrame(
        {
            "image_path": image_paths,
            "features": np.array([[2.0, 0.0], [0.0, 2.0]]),
        }
    )

    dataset.save_features(feature_df, CONFIG)

    with pytest.raises(
        ValueError, match="Features for 'dinov2/defaults' are already present"
    ):
        # rejects on second time
        dataset.save_features(feature_df, CONFIG)


def test_save_features_overwrite(tmp_path: Path, tmp_image_dir: Path):
    dataset = ImageDataset.create(
        image_dir=tmp_image_dir, storage_path=tmp_image_dir / "data.h5", max_images=2
    )
    image_paths = [str(p) for p in dataset.get_image_paths()]
    feature_df = pl.DataFrame(
        {
            "image_path": image_paths,
            "features": np.array([[2.0, 0.0], [0.0, 2.0]]),
        }
    )
    dataset.save_features(feature_df, CONFIG)
    # add derived results that should be removed on overwrite
    dataset.save_clusters(np.array([0, 1]), 2, CONFIG)

    new_df = feature_df.with_columns(features=np.array([[5.0, 5.0], [6.0, 6.0]]))
    dataset.save_features(new_df, CONFIG, overwrite=True)

    saved = dataset.load_data(CONFIG, features=True)
    assert saved["features"].to_numpy().tolist() == [[5.0, 5.0], [6.0, 6.0]]
    assert (
        "clusters"
        # currently reporting as named in h5, i.e. model/opts
        not in dataset.info()["models"][dataset.schema.model_group(CONFIG)]
    )


def test_save_features_unknown_paths(tmp_path: Path, tmp_image_dir: Path):
    dataset = ImageDataset.create(
        image_dir=tmp_image_dir, storage_path=tmp_image_dir / "data.h5", max_images=2
    )
    feature_df = pl.DataFrame(
        {
            "image_path": ["/not/an/image.jpg", "/also/not.jpg"],
            "features": np.array([[2.0, 0.0], [0.0, 2.0]]),
        }
    )
    with pytest.raises(ValueError, match=r"\(2 unknown paths\)"):
        dataset.save_features(feature_df, CONFIG)


def test_save_features_rejects_out_of_order_paths(tmp_path: Path, tmp_image_dir: Path):
    dataset = ImageDataset.create(
        image_dir=tmp_image_dir, storage_path=tmp_path / "data.h5", max_images=2
    )
    image_paths = [str(p) for p in dataset.get_image_paths()]
    feature_df = pl.DataFrame(
        {
            "image_path": list(reversed(image_paths)),
            "features": np.array([[2.0, 0.0], [0.0, 2.0]]),
        }
    )
    with pytest.raises(ValueError, match="stored dataset order"):
        dataset.save_features(feature_df, CONFIG)
    assert not dataset.has_features(CONFIG)


def test_save_features_rejects_out_of_order_subset(tmp_path: Path, tmp_image_dir: Path):
    dataset = ImageDataset.create(
        image_dir=tmp_image_dir, storage_path=tmp_path / "data.h5"
    )
    image_paths = [str(p) for p in dataset.get_image_paths()]
    assert len(image_paths) >= 3
    # subset with one image missing, out of order: must not prune
    subset = [image_paths[2], image_paths[0]]
    feature_df = pl.DataFrame(
        {"image_path": subset, "features": np.array([[2.0, 0.0], [0.0, 2.0]])}
    )
    with pytest.raises(ValueError, match="stored dataset order"):
        dataset.save_features(feature_df, CONFIG)
    # inventory untouched
    assert dataset.image_count == len(image_paths)


def test_save_features_rejects_duplicate_paths(tmp_path: Path, tmp_image_dir: Path):
    dataset = ImageDataset.create(
        image_dir=tmp_image_dir, storage_path=tmp_path / "data.h5", max_images=2
    )
    first = str(next(dataset.get_image_paths()))
    feature_df = pl.DataFrame(
        {"image_path": [first, first], "features": np.array([[2.0, 0.0], [0.0, 2.0]])}
    )
    with pytest.raises(ValueError, match="must be unique"):
        dataset.save_features(feature_df, CONFIG)


def test_has_features(tmp_path: Path, tmp_image_dir: Path):
    dataset = ImageDataset.create(
        image_dir=tmp_image_dir, storage_path=tmp_image_dir / "data.h5", max_images=2
    )
    # use the paths the dataset found; filesystem glob order varies by platform
    image_paths = [str(p) for p in dataset.get_image_paths()]
    feature_df = pl.DataFrame(
        {
            "image_path": image_paths,
            "features": np.array([[2.0, 0.0], [0.0, 2.0]]),
        }
    )
    assert not dataset.has_features(CONFIG)
    dataset.save_features(feature_df, CONFIG)
    assert dataset.has_features(CONFIG)
    assert not dataset.has_features(CLIP_CONFIG)


## test find images utility method


def test_find_images(tmp_image_dir: Path):
    items = list(find_images(tmp_image_dir))
    assert len(items) == 3
    for path in items:
        assert isinstance(path, Path)
        assert path.suffix == ".jpg"


def test_find_images_empty_dir(tmp_path: Path):
    assert list(find_images(tmp_path)) == []


def test_find_images_skips_non_image(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("hello")
    Image.new("RGB", (8, 8)).save(tmp_path / "photo.jpg")
    items = list(find_images(tmp_path))
    assert len(items) == 1


def test_find_images_custom_extensions(tmp_path: Path):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.jpg")
    Image.new("RGB", (8, 8)).save(tmp_path / "b.png")
    items = list(find_images(tmp_path, extensions={".png"}))
    assert len(items) == 1
    assert items[0].suffix == ".png"


def test_find_images_recurses_subdirs(tmp_path: Path):
    sub = tmp_path / "sub"
    sub.mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "top.jpg")
    Image.new("RGB", (8, 8)).save(sub / "nested.jpg")
    items = find_images(tmp_path)
    assert len(list(items)) == 2


def test_find_images_max(tmp_image_dir: Path):
    items = list(find_images(tmp_image_dir, max_images=2))
    assert len(items) == 2
    # which images are returned depends on filesystem order (differs between
    # macOS and Linux), so only check that they are a subset of all images
    assert set(items) <= set(find_images(tmp_image_dir))


def test_find_images_default_extensions(tmp_path: Path):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.jpg")
    Image.new("RGB", (8, 8)).save(tmp_path / "b.png")
    Image.new("RGB", (8, 8)).save(tmp_path / "c.jpeg")
    # Discovery only checks suffixes, so these do not need to be decodable
    # images (which keeps the test independent of optional AVIF support in Pillow).
    (tmp_path / "d.webp").touch()
    (tmp_path / "e.avif").touch()
    dataset = find_images(tmp_path)
    assert len(list(dataset)) == 5


def test_find_images_extension_case_insensitive(tmp_path: Path):
    Image.new("RGB", (8, 8)).save(tmp_path / "upper.JPG", format="JPEG")
    items = find_images(tmp_path)
    assert len(list(items)) == 1


@pytest.mark.parametrize("kwargs", [{"features": True}, {"umap": True}])
def test_load_data_requires_model(tmp_path: Path, tmp_image_dir: Path, kwargs):
    dataset = ImageDataset.create(
        storage_path=tmp_path / "data.h5", image_dir=tmp_image_dir
    )
    with pytest.raises(ValueError, match="model configuration is required"):
        dataset.load_data(**kwargs)


def test_load_data_umap_missing(tmp_path: Path, tmp_image_dir: Path):
    dataset = ImageDataset.create(
        storage_path=tmp_path / "data.h5", image_dir=tmp_image_dir
    )
    df = dataset.get_image_data().with_columns(
        features=pl.Series(np.zeros((dataset.image_count, 4), dtype=np.float32))
    )
    dataset.save_features(df.select("image_path", "features"), CONFIG)
    with pytest.raises(ValueError, match="get_or_compute_umap"):
        dataset.load_data(CONFIG, umap=True)


def test_save_umap(tmp_path: Path, tmp_image_dir: Path):
    dataset = ImageDataset.create(
        storage_path=tmp_path / "data.h5", image_dir=tmp_image_dir
    )
    n = dataset.image_count
    df = dataset.get_image_data().with_columns(
        features=pl.Series(np.zeros((n, 4), dtype=np.float32))
    )
    dataset.save_features(df.select("image_path", "features"), CONFIG)
    assert not dataset.has_umap(CONFIG)
    assert list(dataset.configs_by_label()) == [CONFIG.label]
    assert dataset.configs_by_label(require_umap=True) == {}
    coords = np.random.default_rng(0).random((n, 2))
    dataset.save_umap(coords, CONFIG)
    assert dataset.has_umap(CONFIG)
    assert list(dataset.configs_by_label(require_umap=True)) == [CONFIG.label]
    loaded = dataset.load_data(CONFIG, umap=True)
    assert np.allclose(loaded["umap"].to_numpy(), coords)


## embedding configuration / preprocessing options


def _two_image_dataset(tmp_image_dir: Path) -> tuple[ImageDataset, pl.DataFrame]:
    dataset = ImageDataset.create(
        image_dir=tmp_image_dir, storage_path=tmp_image_dir / "data.h5", max_images=2
    )
    feature_df = pl.DataFrame(
        {
            "image_path": [str(p) for p in dataset.get_image_paths()],
            "features": np.array([[2.0, 0.0], [0.0, 2.0]]),
        }
    )
    return dataset, feature_df


@pytest.mark.parametrize(
    "config, absolute, expected",
    [
        (CONFIG, False, "dinov2/defaults"),
        (CONFIG, True, "image/models/dinov2/defaults"),
        (
            EmbeddingConfig(DEFAULT_MODEL, center_crop=0.6, grayscale=True),
            True,
            "image/models/dinov2/center_crop=0.6_grayscale",
        ),
    ],
)
def test_schema_model_group(config, absolute, expected):
    assert StorageSchema.model_group(config, absolute=absolute) == expected


def test_has_features_any(tmp_image_dir: Path):
    dataset, feature_df = _two_image_dataset(tmp_image_dir)
    # no models group yet
    assert not dataset.has_features()
    dataset.save_features(feature_df, CLIP_CONFIG)
    assert dataset.has_features()


def test_has_features_any_ignores_groups_without_features(tmp_image_dir: Path):
    dataset, _ = _two_image_dataset(tmp_image_dir)
    with h5py.File(dataset.storage_path, "r+") as h5_file:
        h5_file.require_group(dataset.schema.model_group(CONFIG, absolute=True))
    assert not dataset.has_features()


def test_has_features_preprocess_specific(tmp_image_dir: Path):
    dataset, feature_df = _two_image_dataset(tmp_image_dir)
    gray_config = EmbeddingConfig(DEFAULT_MODEL, grayscale=True)
    dataset.save_features(feature_df, gray_config)
    assert dataset.has_features(gray_config)
    # same model with different preprocessing is a different feature set
    assert not dataset.has_features(CONFIG)
    assert not dataset.has_features(
        EmbeddingConfig(DEFAULT_MODEL, center_crop=0.6, grayscale=True)
    )


def test_save_load_features_per_preprocess_config(tmp_image_dir: Path):
    dataset, feature_df = _two_image_dataset(tmp_image_dir)
    crop_config = EmbeddingConfig(DEFAULT_MODEL, center_crop=0.6)
    dataset.save_features(feature_df, CONFIG)
    # same model with different options does not conflict with existing features
    dataset.save_features(
        feature_df.with_columns(features=np.array([[7.0, 7.0], [8.0, 8.0]])),
        crop_config,
    )

    default_df = dataset.load_data(CONFIG, features=True)
    crop_df = dataset.load_data(crop_config, features=True)
    assert default_df["features"].to_numpy().tolist() == [[2.0, 0.0], [0.0, 2.0]]
    assert crop_df["features"].to_numpy().tolist() == [[7.0, 7.0], [8.0, 8.0]]

    with h5py.File(dataset.storage_path, "r") as h5_file:
        assert set(h5_file["image/models/dinov2"]) == {"defaults", "center_crop=0.6"}
    assert set(dataset.info()["models"]) == {
        "dinov2/defaults",
        "dinov2/center_crop=0.6",
    }


def test_overwrite_only_replaces_matching_config(tmp_image_dir: Path):
    dataset, feature_df = _two_image_dataset(tmp_image_dir)
    gray_config = EmbeddingConfig(DEFAULT_MODEL, grayscale=True)
    dataset.save_features(feature_df, CONFIG)
    dataset.save_features(feature_df, gray_config)
    dataset.save_clusters(np.array([0, 1]), 2, CONFIG)

    dataset.save_features(
        feature_df.with_columns(features=np.ones((2, 2))), gray_config, overwrite=True
    )
    # default config features and derived clusters are untouched
    df = dataset.load_data(CONFIG, features=True, clusters=True)
    assert df["features"].to_numpy().tolist() == [[2.0, 0.0], [0.0, 2.0]]
    assert df["cluster_kmeans_2"].to_list() == [0, 1]


def test_save_clusters_preprocess_config(tmp_image_dir: Path):
    dataset, feature_df = _two_image_dataset(tmp_image_dir)
    gray_config = EmbeddingConfig(DEFAULT_MODEL, grayscale=True)
    dataset.save_features(feature_df, CONFIG)
    dataset.save_features(feature_df, gray_config)
    dataset.save_clusters(np.array([1, 0]), 2, gray_config)

    # clusters stored with the matching config only
    gray_df = dataset.load_data(gray_config, features=True, clusters=True)
    assert gray_df["cluster_kmeans_2"].to_list() == [1, 0]
    default_df = dataset.load_data(CONFIG, features=True, clusters=True)
    assert "cluster_kmeans_2" not in default_df.columns
    info = dataset.info()["models"]
    assert info["dinov2/grayscale"]["clusters"] == [
        {"method": "kmeans", "n_clusters": 2, "size": 2}
    ]
    assert "clusters" not in info["dinov2/defaults"]


def test_load_data_clusters_by_k(tmp_image_dir: Path):
    dataset, feature_df = _two_image_dataset(tmp_image_dir)
    dataset.save_features(feature_df, CONFIG)
    dataset.save_clusters(np.array([1, 0]), 2, CONFIG)
    dataset.save_clusters(np.array([0, 0]), 1, CONFIG)

    df = dataset.load_data(CONFIG, clusters=True)
    assert {"cluster_kmeans_1", "cluster_kmeans_2"} <= set(df.columns)
    df = dataset.load_data(CONFIG, clusters=2)
    assert "cluster_kmeans_2" in df.columns
    assert "cluster_kmeans_1" not in df.columns
    with pytest.raises(ValueError, match="configuration is required"):
        dataset.load_data(clusters=True)


def test_save_features_stores_config_attrs(tmp_path: Path, tmp_image_dir: Path):
    storage_path = tmp_path / "data.h5"
    dataset = ImageDataset.create(
        storage_path=storage_path, image_dir=tmp_image_dir, max_images=2
    )
    paths = [str(p) for p in dataset.get_image_paths()]
    df = pl.DataFrame(
        {"image_path": paths, "features": np.array([[1.0, 0.0], [0.0, 1.0]])}
    )
    custom = EmbeddingConfig(DEFAULT_MODEL, center_crop=0.6, grayscale=True)
    dataset.save_features(df, CONFIG)
    dataset.save_features(df, custom)

    with h5py.File(storage_path, "r") as h5_file:
        attrs = h5_file[dataset.schema.model_group(custom, absolute=True)].attrs
        assert attrs["model"] == DEFAULT_MODEL.name
        assert attrs["model_id"] == DEFAULT_MODEL.id
        assert attrs["center_crop"] == 0.6
        assert attrs["grayscale"]
        default_attrs = h5_file[dataset.schema.model_group(CONFIG, absolute=True)].attrs
        assert "center_crop" not in default_attrs
        assert not default_attrs["grayscale"]

    configs = dataset.get_configs()
    assert set(configs) == {CONFIG, custom}
    assert all(isinstance(c.grayscale, bool) for c in configs)

    # keyed by label, default preprocessing first
    by_label = dataset.configs_by_label()
    assert list(by_label) == [CONFIG.label, custom.label]
    assert by_label[custom.label] == custom

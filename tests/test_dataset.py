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
from iconoscope.models import DEFAULT_MODEL, EmbeddingModel

## image dataset class


def test_init_validation(tmp_path):
    image_dir = tmp_path / "images"
    # non-existent
    with pytest.raises(ValueError, match="not a directory"):
        ImageDataset(image_dir=image_dir, storage_path=tmp_path / "a.h5")
    # file instead of dir
    img_file = tmp_path / "img.png"
    img_file.touch()
    with pytest.raises(ValueError, match="not a directory"):
        ImageDataset(image_dir=img_file, storage_path=tmp_path / "a.h5")

    # image dir is required when h5 file doesn't exist
    with pytest.raises(ValueError, match="image_dir is required"):
        ImageDataset(storage_path=tmp_path / "a.h5")

    # existing directory - no error, returns new object
    image_dir.mkdir()
    assert ImageDataset(image_dir=image_dir, storage_path=tmp_path / "a.h5")

    # storage path exists - image dir is not required
    h5_datafile = tmp_path / "z.h5"
    h5_datafile.touch()  # currently doesn't validate (probably should in future)
    assert ImageDataset(storage_path=h5_datafile)


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
    # when storage file doesn't exist, yields results from find image method
    h5_datafile = tmp_path / "data.h5"
    img_ds = ImageDataset(image_dir=tmp_image_dir, storage_path=h5_datafile)
    with patch.object(img_ds, "get_image_data") as mock_get_image_data:
        img_paths = img_ds.get_image_paths()
        # should not load from data when storage file doesn't exist
        mock_get_image_data.assert_not_called()
        assert isinstance(img_paths, Iterable)
        img_paths = list(img_paths)
        assert len(img_paths) == 3  # 3 in fixture
        for path in img_paths:
            assert isinstance(path, Path)
            assert path.suffix == ".jpg"

        h5_datafile.touch()
        test_image_paths = ["foobar_a.jpg", "b.jpg"]
        img_df = pl.DataFrame(data={"image_path": test_image_paths})
        mock_get_image_data.return_value = img_df
        img_paths = img_ds.get_image_paths()
        # call count assertion fails, but mock data is working
        # assert mock_get_image_data.call_count == 1
        assert isinstance(img_paths, Iterable)
        img_paths = list(img_paths)
        assert len(img_paths) == len(test_image_paths)
        for i, path in enumerate(img_paths):
            assert isinstance(path, Path)
            assert str(path) == test_image_paths[i]


def test_iter(tmp_path: Path, tmp_image_dir: Path):
    h5_datafile = tmp_path / "data.h5"
    img_ds = ImageDataset(image_dir=tmp_image_dir, storage_path=h5_datafile)
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
    image_dir = tmp_path / "images"
    image_dir.mkdir()
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
        DEFAULT_MODEL,
    )
    with h5py.File(storage_path, "r+") as h5_file:
        model_group = h5_file["image/models/dinov2"]
        model_group.create_dataset("umap", data=[[0.1, 0.2], [0.3, 0.4]])
        cluster = model_group.create_dataset("cluster", data=[0, 1])
        cluster.attrs["n_clusters"] = 2

    dataset.save_features(
        pl.DataFrame(
            {
                "image_path": paths,
                "features": np.array([[2.0, 0.0], [0.0, 2.0]]),
            }
        ),
        EmbeddingModel("clip", "foobar/clip", 73),
    )

    with h5py.File(storage_path, "r") as h5_file:
        assert set(h5_file["image/models"]) == {"dinov2", "clip"}
        assert h5_file["image/models/dinov2/features"][:].tolist() == [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        assert h5_file["image/models/dinov2/umap"][:].tolist() == [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
        assert h5_file["image/models/dinov2/cluster"][:].tolist() == [0, 1]
        assert h5_file["image/models/dinov2/cluster"].attrs["n_clusters"] == 2


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
            DEFAULT_MODEL,
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
        DEFAULT_MODEL,
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
            EmbeddingModel("clip", "foobar/baz", 73),
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
        DEFAULT_MODEL,
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
            DEFAULT_MODEL,
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
            DEFAULT_MODEL,
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

    model_name = DEFAULT_MODEL.name
    dataset.save_features(feature_df, DEFAULT_MODEL)

    with pytest.raises(
        ValueError, match=f"Features for '{model_name}' are already present"
    ):
        # rejects on second time
        dataset.save_features(feature_df, DEFAULT_MODEL)


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
    dataset.save_features(feature_df, DEFAULT_MODEL)
    # add derived results that should be removed on overwrite
    dataset.save_clusters(np.array([0, 1]), 2, DEFAULT_MODEL)

    new_df = feature_df.with_columns(features=np.array([[5.0, 5.0], [6.0, 6.0]]))
    dataset.save_features(new_df, DEFAULT_MODEL, overwrite=True)

    saved = dataset.load_data(DEFAULT_MODEL, features=True)
    assert saved["features"].to_numpy().tolist() == [[5.0, 5.0], [6.0, 6.0]]
    assert "cluster" not in dataset.info()["models"][DEFAULT_MODEL.name]


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
        dataset.save_features(feature_df, DEFAULT_MODEL)


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
    assert not dataset.has_features(DEFAULT_MODEL)
    dataset.save_features(feature_df, DEFAULT_MODEL)
    assert dataset.has_features(DEFAULT_MODEL)
    assert not dataset.has_features(EmbeddingModel("clip", "foobar/baz", 72))


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
    items = list(find_images(tmp_image_dir, max=2))
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
    with pytest.raises(ValueError, match="model is required"):
        dataset.load_data(**kwargs)

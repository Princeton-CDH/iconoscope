from pathlib import Path

import h5py
import numpy as np
import pytest

from iconoscope.models import DEFAULT_MODEL, EmbeddingConfig
from iconoscope.store import DatasetStore, StorageSchema

PATHS = ["a.jpg", "sub/b.png"]
SIZES = [(32, 24), (10, 20)]


@pytest.fixture
def store(tmp_path: Path) -> DatasetStore:
    return DatasetStore.create(tmp_path / "data.h5", tmp_path / "images", PATHS, SIZES)


def test_init_requires_existing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        DatasetStore(tmp_path / "missing.h5")


def test_create_writes_schema_and_images(tmp_path: Path, store: DatasetStore):
    with h5py.File(store.path, "r") as f:
        assert f.attrs["schema"] == StorageSchema.schema_name
        assert f.attrs["schema_version"] == StorageSchema.version
        img_grp = f["image"]
        assert img_grp.attrs["image_dir"] == str(tmp_path / "images")
        assert img_grp["paths"][:].astype("T").tolist() == PATHS
        assert img_grp["sizes"][:].tolist() == [list(s) for s in SIZES]
        assert img_grp["paths"].compression == "gzip"


def test_create_validation(tmp_path: Path, store: DatasetStore):
    with pytest.raises(ValueError, match="already exists"):
        DatasetStore.create(store.path, tmp_path, PATHS, SIZES)
    with pytest.raises(ValueError, match="must be absolute"):
        DatasetStore.create(tmp_path / "x.h5", Path("images"), PATHS, SIZES)
    with pytest.raises(ValueError, match="same length"):
        DatasetStore.create(tmp_path / "y.h5", tmp_path, PATHS, SIZES[:1])


def test_image_reads(tmp_path: Path, store: DatasetStore):
    assert store.image_dir == tmp_path / "images"
    assert store.image_count == 2
    df = store.image_data()
    assert df.columns == ["image_path", "image_size", "aspect_ratio"]
    assert df["image_path"].to_list() == [str(tmp_path / "images" / p) for p in PATHS]
    assert df["image_size"].to_list() == [list(s) for s in SIZES]
    assert df["aspect_ratio"].to_list() == [32 / 24, 0.5]


def test_open_reuses_file(store: DatasetStore):
    with store.open() as opened:
        assert opened is store
        h5_file = store._file
        assert h5_file is not None
        # nested calls reuse the open file
        store.image_data()
        with store.open():
            assert store._file is h5_file
    assert store._file is None


def test_open_mode_validation(store: DatasetStore):
    with pytest.raises(ValueError, match="Unsupported mode"):
        with store.open("w"):
            pass
    with store.open("r"):
        with pytest.raises(ValueError, match="read-only"):
            with store.open("r+"):
                pass
    # nested read inside a writable open is fine
    with store.open("r+"):
        with store.open("r"):
            assert store._file.mode == "r+"


def test_open_closes_on_error(store: DatasetStore):
    with pytest.raises(RuntimeError):
        with store.open():
            raise RuntimeError
    assert store._file is None


def test_configs(store: DatasetStore):
    assert store.configs() == []
    custom = EmbeddingConfig(DEFAULT_MODEL, center_crop=0.6, grayscale=True)
    with h5py.File(store.path, "r+") as f:
        grp = f.create_group(StorageSchema.model_group(custom, absolute=True))
        grp.attrs["model"] = DEFAULT_MODEL.name
        grp.attrs["center_crop"] = 0.6
        grp.attrs["grayscale"] = True
        grp.create_dataset("features", data=[[1.0, 0.0], [0.0, 1.0]])
        # group without features is skipped
        f.create_group(
            StorageSchema.model_group(EmbeddingConfig(DEFAULT_MODEL), absolute=True)
        )
        # unknown model is skipped
        unknown = f.create_group(f"{StorageSchema.all_models_path()}/nope/defaults")
        unknown.attrs["model"] = "nope"
        unknown.create_dataset("features", data=[[1.0]])

    configs = store.configs()
    assert configs == [custom]
    assert isinstance(configs[0].grayscale, bool)


CONFIG = EmbeddingConfig(DEFAULT_MODEL)
FEATURES = np.array([[1.0, 0.0], [0.0, 1.0]])


def test_write_and_read_features(store: DatasetStore):
    assert not store.has_features()
    assert not store.has_features(CONFIG)
    store.write_features(CONFIG, FEATURES)
    assert store.has_features()
    assert store.has_features(CONFIG)
    assert store.configs() == [CONFIG]
    assert store.features(CONFIG).tolist() == FEATURES.tolist()
    assert store.umap(CONFIG) is None
    assert store.clusters(CONFIG) is None


def test_write_features_validation(store: DatasetStore):
    with pytest.raises(ValueError, match="2-dimensional"):
        store.write_features(CONFIG, np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="dataset has 2 images"):
        store.write_features(CONFIG, FEATURES[:1])
    store.write_features(CONFIG, FEATURES)
    with pytest.raises(ValueError, match="already present"):
        store.write_features(CONFIG, FEATURES)


def test_write_features_overwrite_removes_derived(store: DatasetStore):
    store.write_features(CONFIG, FEATURES)
    store.write_umap(CONFIG, np.array([[0.1, 0.2], [0.3, 0.4]]))
    store.write_clusters(CONFIG, np.array([0, 1]), 2)
    store.write_features(CONFIG, FEATURES * 2, overwrite=True)
    assert store.features(CONFIG).tolist() == (FEATURES * 2).tolist()
    assert store.umap(CONFIG) is None
    assert store.clusters(CONFIG) is None


def test_write_umap_and_clusters(store: DatasetStore):
    with pytest.raises(ValueError, match="No features"):
        store.write_umap(CONFIG, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="No features"):
        store.write_clusters(CONFIG, np.array([0, 1]), 2)
    store.write_features(CONFIG, FEATURES)
    store.write_umap(CONFIG, np.array([[0.1, 0.2], [0.3, 0.4]]))
    store.write_clusters(CONFIG, np.array([0, 1]), 2)
    # replacing existing results is allowed
    store.write_clusters(CONFIG, np.array([1, 1]), 1)
    assert store.umap(CONFIG).tolist() == [[0.1, 0.2], [0.3, 0.4]]
    assert store.clusters(CONFIG).tolist() == [1, 1]
    with h5py.File(store.path, "r") as f:
        cluster_ds = f[StorageSchema.model_group(CONFIG, absolute=True)]["cluster"]
        assert cluster_ds.attrs["n_clusters"] == 1


def test_prune_images(tmp_path: Path, store: DatasetStore):
    keep = str(tmp_path / "images" / "sub/b.png")
    assert store.image_count == 2
    with pytest.raises(ValueError, match="not in the dataset"):
        store.prune_images([keep, "/elsewhere/c.jpg"])
    store.prune_images([keep])
    # cached count is cleared
    assert store.image_count == 1
    df = store.image_data()
    assert df["image_path"].to_list() == [keep]
    assert df["image_size"].to_list() == [[10, 20]]
    store.write_features(CONFIG, FEATURES[:1])
    with pytest.raises(ValueError, match="existing models are present"):
        store.prune_images([])


def test_repr_and_equality(store: DatasetStore):
    assert repr(store) == f"DatasetStore(path={store.path!r})"
    # string paths are converted; equality is by path only
    other = DatasetStore(str(store.path))
    assert other.path == store.path
    with store.open():
        assert store == other

"""HDF5 storage for iconoscope image datasets.

:class:`DatasetStore` is the only code that should need to know the HDF5
layout. It has no torch or umap dependencies, so stored data can be loaded
and explored without them.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Self

import h5py
import numpy as np
import polars as pl

from iconoscope.models import AVAILABLE_MODELS, EmbeddingConfig


class StorageSchema:
    """
    Group, dataset, and attribute names for an iconoscope image dataset
    within an HDF5 file.
    """

    # schema versioning
    schema_name: str = "iconoscope"
    version: float = 0.2

    image_group: str = "image"
    # relative to image group
    image_ds_paths: str = "paths"
    image_ds_sizes: str = "sizes"
    all_models_group: str = "models"
    # relative to model group
    features_ds: str = "features"
    umap_ds: str = "umap"
    clusters_group: str = "clusters"

    # schema attrs (file root)
    attr_schema = "schema"
    attr_schema_version = "schema_version"
    # image group attrs
    attr_image_dir = "image_dir"
    # model group attrs
    attr_model = "model"
    attr_model_id = "model_id"
    # cluster dataset attrs
    attr_cluster_method = "method"
    attr_n_clusters = "n_clusters"

    #: supported clustering methods
    cluster_methods: tuple[str, ...] = ("kmeans",)

    @classmethod
    def all_models_path(cls) -> str:
        """Absolute path of the group containing all model groups."""
        return f"{cls.image_group}/{cls.all_models_group}"

    @classmethod
    def model_group(cls, config: EmbeddingConfig, absolute: bool = False) -> str:
        """Group path for features and derived data for a model + preprocessing
        configuration; relative to the all-models group unless `absolute`."""
        base_path = f"{cls.all_models_path()}/" if absolute else ""
        return f"{base_path}{config.model.name}/{config.key}"

    @classmethod
    def cluster_name(cls, method: str, n_clusters: int) -> str:
        """Dataset name (within the clusters group) for cluster labels
        generated with a method and options, e.g. ``kmeans_5``."""
        if method not in cls.cluster_methods:
            raise ValueError(
                f"Unsupported cluster method `{method}`; "
                f"supported: {', '.join(cls.cluster_methods)}"
            )
        return f"{method}_{n_clusters}"


#: compression used for all datasets
COMPRESSION = "gzip"


class InvalidDatasetError(ValueError):
    """File is not a valid iconoscope dataset (not HDF5, wrong schema or
    version, or missing required data)."""


def _to_python(value):
    """Convert numpy scalars (as returned for HDF5 attrs) to python types."""
    return value.item() if hasattr(value, "item") else value


@dataclass
class DatasetStore:
    """Read and write access to an existing iconoscope HDF5 dataset file.

    Methods open the file as needed. To reuse a single open file across
    several calls, use :meth:`open` as a context manager::

        with store.open("r"):
            df = store.image_data()
            configs = store.configs()

    Use :meth:`create` to create a new dataset file.
    """

    #: path to an existing dataset file
    path: Path

    #: group, dataset, and attribute names within the file
    schema: ClassVar[type[StorageSchema]] = StorageSchema

    #: file handle while open via :meth:`open`; runtime state only
    _file: h5py.File | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset `{self.path}` does not exist")
        self.validate()

    def validate(self) -> None:
        """Minimal check that the file is an iconoscope dataset this version
        can read: an HDF5 file with the expected schema name and version, and
        the required image group, datasets, and attributes. Does not check
        model data. Raises :class:`InvalidDatasetError` if not valid."""
        if not h5py.is_hdf5(self.path):
            raise InvalidDatasetError(f"`{self.path}` is not an HDF5 file")
        with self._h5() as f:
            schema = _to_python(f.attrs.get(self.schema.attr_schema))
            if schema != self.schema.schema_name:
                raise InvalidDatasetError(
                    f"`{self.path}` is not an {self.schema.schema_name} dataset "
                    f"(schema: {schema})"
                )
            version = _to_python(f.attrs.get(self.schema.attr_schema_version))
            if version != self.schema.version:
                raise InvalidDatasetError(
                    f"`{self.path}` has unsupported schema version {version}; "
                    f"expected {self.schema.version}"
                )
            img_grp = f.get(self.schema.image_group)
            if not isinstance(img_grp, h5py.Group):
                raise InvalidDatasetError(
                    f"`{self.path}` is missing required group "
                    f"`{self.schema.image_group}`"
                )
            for ds_name in (self.schema.image_ds_paths, self.schema.image_ds_sizes):
                if not isinstance(img_grp.get(ds_name), h5py.Dataset):
                    raise InvalidDatasetError(
                        f"`{self.path}` is missing required dataset "
                        f"`{self.schema.image_group}/{ds_name}`"
                    )
            if self.schema.attr_image_dir not in img_grp.attrs:
                raise InvalidDatasetError(
                    f"`{self.path}` is missing required attribute "
                    f"`{self.schema.attr_image_dir}` on `{self.schema.image_group}`"
                )
            all_models_grp = f.get(self.schema.all_models_path())
            if all_models_grp is not None and not isinstance(
                all_models_grp, h5py.Group
            ):
                raise InvalidDatasetError(
                    f"`{self.path}`: `{self.schema.all_models_path()}` is not a group"
                )

    ## file access

    @contextmanager
    def open(self, mode: str = "r") -> Iterator[Self]:
        """Keep the file open for the duration of the block; methods called
        within the block reuse the open file. Mode is ``"r"`` or ``"r+"``.
        Nested calls reuse the outer file, which must be writable if
        ``"r+"`` is requested."""
        if mode not in ("r", "r+"):
            raise ValueError(f"Unsupported mode `{mode}`; use 'r' or 'r+'")
        if self._file is not None:
            if mode == "r+" and self._file.mode != "r+":
                raise ValueError("Store is already open read-only")
            yield self
            return
        with h5py.File(self.path, mode) as h5_file:
            self._file = h5_file
            try:
                yield self
            finally:
                self._file = None

    @property
    def file(self) -> h5py.File:
        """The open file handle; only available within :meth:`open`."""
        if self._file is None:
            raise RuntimeError("Store is not open; use `with store.open():`")
        return self._file

    @contextmanager
    def _h5(self, mode: str = "r") -> Iterator[h5py.File]:
        """Yield the open file, opening it for this call if needed."""
        with self.open(mode):
            yield self._file

    @classmethod
    def _write_dataset(cls, group: h5py.Group, name: str, data) -> h5py.Dataset:
        """Create (or replace) a compressed dataset within a group."""
        if name in group:
            del group[name]
        return group.create_dataset(name, data=data, compression=COMPRESSION)

    @classmethod
    def _write_image_group(
        cls,
        h5_file: h5py.File,
        image_dir: Path,
        rel_paths: Sequence[str],
        sizes: Sequence[tuple[int, int]],
    ) -> None:
        """Write image directory, paths, and sizes to the image group."""
        if len(rel_paths) != len(sizes):
            raise ValueError("image paths and sizes must be the same length")
        img_grp = h5_file.require_group(cls.schema.image_group)
        img_grp.attrs[cls.schema.attr_image_dir] = str(image_dir)
        cls._write_dataset(img_grp, cls.schema.image_ds_paths, list(rel_paths))
        cls._write_dataset(img_grp, cls.schema.image_ds_sizes, list(sizes))

    ## creation

    @classmethod
    def create(
        cls,
        path: Path | str,
        image_dir: Path,
        rel_paths: Sequence[str],
        sizes: Sequence[tuple[int, int]],
    ) -> Self:
        """Create a new dataset file with schema metadata and image inventory.
        `image_dir` must be absolute; `rel_paths` are relative to it and are
        stored in the order given."""
        path = Path(path)
        image_dir = Path(image_dir)
        if path.exists():
            raise ValueError(f"Dataset already exists: {path}")
        if not image_dir.is_absolute():
            raise ValueError(f"Image directory `{image_dir}` must be absolute")
        # write to a new file; mode "x" fails if it was created in the meantime
        with h5py.File(path, "x") as h5_file:
            h5_file.attrs[cls.schema.attr_schema] = cls.schema.schema_name
            h5_file.attrs[cls.schema.attr_schema_version] = cls.schema.version
            cls._write_image_group(h5_file, image_dir, rel_paths, sizes)
        return cls(path)

    ## reads

    # image group is only written on creation, so these are safe to cache

    @cached_property
    def image_dir(self) -> Path:
        """Absolute base directory for the images in this dataset."""
        with self._h5() as f:
            return Path(f[self.schema.image_group].attrs[self.schema.attr_image_dir])

    @cached_property
    def image_count(self) -> int:
        """Number of images in the dataset."""
        with self._h5() as f:
            return f[self.schema.image_group][self.schema.image_ds_paths].size

    def image_paths(self) -> list[Path]:
        """Absolute image paths, in stored order (without sizes)."""
        with self._h5() as f:
            img_grp = f[self.schema.image_group]
            image_dir = Path(img_grp.attrs[self.schema.attr_image_dir])
            rel_paths = img_grp[self.schema.image_ds_paths][:].astype("T")
        return [image_dir / p for p in rel_paths]

    def image_data(self) -> pl.DataFrame:
        """Image inventory, in stored order: absolute ``image_path``,
        ``image_size`` (width, height), and ``aspect_ratio``."""
        with self._h5() as f:
            img_grp = f[self.schema.image_group]
            image_dir = Path(img_grp.attrs[self.schema.attr_image_dir])
            rel_paths = img_grp[self.schema.image_ds_paths][:].astype("T")
            sizes = img_grp[self.schema.image_ds_sizes][:]
        return pl.DataFrame(
            {
                "image_path": [str(image_dir / p) for p in rel_paths],
                "image_size": sizes,
            }
        ).with_columns(
            aspect_ratio=pl.col.image_size.arr.first().truediv(
                pl.col.image_size.arr.last()
            )
        )

    def configs(self) -> list[EmbeddingConfig]:
        """Embedding configurations (model + preprocessing options) with
        features stored in this dataset, reconstructed from group attributes.
        Groups without config attributes or for unavailable models are skipped."""
        configs = []
        with self._h5() as f:
            all_models_grp = f.get(self.schema.all_models_path())
            if all_models_grp is None:
                return configs
            for model_grp in all_models_grp.values():
                for opt_grp in model_grp.values():
                    if self.schema.features_ds not in opt_grp:
                        continue
                    model = AVAILABLE_MODELS.get(
                        opt_grp.attrs.get(self.schema.attr_model)
                    )
                    if model is None:
                        continue
                    opts = {
                        opt: _to_python(opt_grp.attrs[opt])
                        for opt in EmbeddingConfig.option_names()
                        if opt in opt_grp.attrs
                    }
                    configs.append(EmbeddingConfig(model, **opts))
        return configs

    def has_features(self, config: EmbeddingConfig | None = None) -> bool:
        """Whether features are stored for a model + preprocessing
        configuration; if no configuration is given, for any configuration."""
        with self._h5() as f:
            if config is None:
                all_models_grp = f.get(self.schema.all_models_path())
                if all_models_grp is None:
                    return False
                return any(
                    self.schema.features_ds in opt_grp
                    for model_grp in all_models_grp.values()
                    for opt_grp in model_grp.values()
                )
            model_grp = f.get(self.schema.model_group(config, absolute=True))
            return model_grp is not None and self.schema.features_ds in model_grp

    def _model_dataset(self, config: EmbeddingConfig, name: str) -> h5py.Dataset:
        """Dataset within the group for a config; file must be open.
        Raises KeyError if not present."""
        return self.file[self.schema.model_group(config, absolute=True)][name]

    def features(self, config: EmbeddingConfig):
        """Stored features for a configuration as a numpy array."""
        with self.open():
            return self._model_dataset(config, self.schema.features_ds)[:]

    def umap(self, config: EmbeddingConfig):
        """Stored umap coordinates for a configuration, or None."""
        with self.open():
            try:
                return self._model_dataset(config, self.schema.umap_ds)[:]
            except KeyError:
                return None

    def _clusters_group(self, config: EmbeddingConfig) -> h5py.Group | None:
        """Clusters group for a config, or None; file must be open."""
        return self.file.get(
            f"{self.schema.model_group(config, absolute=True)}/{self.schema.clusters_group}"
        )

    @classmethod
    def _cluster_key(cls, ds: h5py.Dataset) -> tuple[str, int]:
        """Clustering method and number of clusters for a stored result."""
        return (
            _to_python(ds.attrs[cls.schema.attr_cluster_method]),
            _to_python(ds.attrs[cls.schema.attr_n_clusters]),
        )

    def cluster_info(self, config: EmbeddingConfig) -> list[dict]:
        """Stored cluster results for a configuration: a list of dicts with
        ``method``, ``n_clusters``, and ``size`` (number of labels), sorted
        by method and number of clusters."""
        with self.open():
            grp = self._clusters_group(config)
            if grp is None:
                return []
            info = []
            for ds in grp.values():
                method, n_clusters = self._cluster_key(ds)
                info.append(
                    {"method": method, "n_clusters": n_clusters, "size": ds.size}
                )
        return sorted(info, key=lambda c: (c["method"], c["n_clusters"]))

    def clusters(
        self,
        config: EmbeddingConfig,
        method: str = "kmeans",
        n_clusters: int | None = None,
    ) -> dict[tuple[str, int], np.ndarray]:
        """Stored cluster labels for a configuration, keyed by
        ``(method, n_clusters)`` and sorted. If `n_clusters` is specified,
        only returns results for that method and number of clusters (empty
        if not present); otherwise returns all stored results."""
        with self.open():
            grp = self._clusters_group(config)
            if grp is None:
                return {}
            if n_clusters is not None:
                name = self.schema.cluster_name(method, n_clusters)
                return {(method, n_clusters): grp[name][:]} if name in grp else {}
            return dict(sorted((self._cluster_key(ds), ds[:]) for ds in grp.values()))

    ## writes

    @staticmethod
    def _check_path_order(
        paths: pl.Series | Sequence[str], stored: pl.Series | Sequence[str]
    ) -> None:
        """Check that `paths` are unique, all in `stored`, and in stored
        order (gaps allowed). Stored data is aligned to image order by row
        position, so this order is fixed when the dataset is created."""
        paths = pl.Series(paths, dtype=pl.String)
        stored = pl.Series(stored, dtype=pl.String)
        if paths.is_duplicated().any():
            raise ValueError("image paths must be unique")
        if not paths.is_in(stored.implode()).all():
            raise ValueError("image paths not in the dataset")
        if not paths.equals(
            stored.filter(stored.is_in(paths.implode())), check_names=False
        ):
            raise ValueError("image paths must be in stored dataset order")

    def prune_images(self, keep_paths: pl.Series | Sequence[str]) -> None:
        """Reduce the image inventory to `keep_paths` (absolute, as returned by
        :meth:`image_data`), which must be in stored order. Only allowed before
        any features are stored, since stored features are aligned to image
        order."""
        with self.open("r+"):
            if self.has_features():
                raise ValueError(
                    "cannot prune image paths when existing models are present"
                )
            img_df = self.image_data()
            keep_paths = pl.Series(keep_paths, dtype=pl.String)
            self._check_path_order(keep_paths, img_df["image_path"])
            keep_df = img_df.filter(pl.col.image_path.is_in(keep_paths.implode()))
            rel_paths = [
                str(Path(p).relative_to(self.image_dir)) for p in keep_df["image_path"]
            ]
            self._write_image_group(
                self.file, self.image_dir, rel_paths, keep_df["image_size"].to_list()
            )
        # image count changed; clear cached value
        self.__dict__.pop("image_count", None)

    def write_features(
        self,
        config: EmbeddingConfig,
        features,
        image_paths: pl.Series | Sequence[str],
        overwrite: bool = False,
    ) -> None:
        """Store features for a configuration, along with config attributes so
        it can be reconstructed. Features must be 2-dimensional with one row
        per image; `image_paths` (absolute, one per row) must match the stored
        images exactly, in stored order. If features for the configuration
        exist, raises ValueError unless `overwrite`, in which case the whole
        model group is replaced (removing derived umap/cluster data)."""
        if features.ndim != 2:
            raise ValueError("features must be 2-dimensional")
        if len(image_paths) != features.shape[0]:
            raise ValueError(
                f"features have {features.shape[0]} rows; "
                f"got {len(image_paths)} image paths"
            )
        with self.open("r+"):
            if features.shape[0] != self.image_count:
                raise ValueError(
                    f"features have {features.shape[0]} rows; "
                    f"dataset has {self.image_count} images"
                )
            self._check_path_order(image_paths, self.image_data()["image_path"])
            all_models_grp = self.file.require_group(self.schema.all_models_path())
            group_name = self.schema.model_group(config)
            if group_name in all_models_grp:
                if not overwrite:
                    raise ValueError(
                        f"Features for '{group_name}' are already present "
                        "and overwrite not requested."
                    )
                del all_models_grp[group_name]
            model_grp = all_models_grp.create_group(group_name)
            model_grp.attrs[self.schema.attr_model] = config.model.name
            model_grp.attrs[self.schema.attr_model_id] = config.model.id
            for opt, value in config.opts.items():
                # HDF5 attrs can't store None; omit unset options
                if value is not None:
                    model_grp.attrs[opt] = value
            self._write_dataset(model_grp, self.schema.features_ds, features)

    def _feature_group(self, config: EmbeddingConfig) -> h5py.Group:
        """Model group for a config, which must have features; file must be open."""
        if not self.has_features(config):
            raise ValueError(f"No features stored for '{config.label}'")
        return self.file[self.schema.model_group(config, absolute=True)]

    def write_umap(self, config: EmbeddingConfig, coords) -> None:
        """Store (or replace) umap coordinates for a configuration."""
        with self.open("r+"):
            self._write_dataset(
                self._feature_group(config), self.schema.umap_ds, coords
            )

    def write_clusters(
        self,
        config: EmbeddingConfig,
        labels,
        n_clusters: int,
        method: str = "kmeans",
    ) -> None:
        """Store (or replace) cluster labels for a configuration, generated
        with the specified method and number of clusters. Results for
        different methods and numbers of clusters are stored separately."""
        name = self.schema.cluster_name(method, n_clusters)
        if len(labels) != self.image_count:
            raise ValueError(
                f"{len(labels)} cluster labels; dataset has {self.image_count} images"
            )
        with self.open("r+"):
            grp = self._feature_group(config).require_group(self.schema.clusters_group)
            ds = self._write_dataset(grp, name, labels)
            ds.attrs[self.schema.attr_cluster_method] = method
            ds.attrs[self.schema.attr_n_clusters] = n_clusters

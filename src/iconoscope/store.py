"""HDF5 storage for iconoscope image datasets.

:class:`DatasetStore` is the only code that should need to know the HDF5
layout. It has no torch or umap dependencies, so stored data can be loaded
and explored without them.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Self

import h5py
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
    cluster_ds: str = "cluster"

    # schema attrs (file root)
    attr_schema = "schema"
    attr_schema_version = "schema_version"
    # image group attrs
    attr_image_dir = "image_dir"
    # model group attrs
    attr_model = "model"
    attr_model_id = "model_id"
    # cluster dataset attrs
    attr_n_clusters = "n_clusters"

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


#: compression used for all datasets
COMPRESSION = "gzip"


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

    def image_dir(self) -> Path:
        """Absolute base directory for the images in this dataset."""
        with self._h5() as f:
            return Path(f[self.schema.image_group].attrs[self.schema.attr_image_dir])

    def image_count(self) -> int:
        """Number of images in the dataset."""
        with self._h5() as f:
            return f[self.schema.image_group][self.schema.image_ds_paths].size

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
